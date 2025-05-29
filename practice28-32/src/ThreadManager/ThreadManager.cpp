#include <iostream>
#include <stdexcept>
#include <fstream>
#include <stdexcept>
#include <vector>

class SDR {
   public:
    SDRcfg::SDRConfig config;
    // TODO: Нужно использовать кольцевые буферы без блокировок который работает
    // на основе атомарных индексов.
    std::unique_ptr<int16_t[]> rxBuffer;
    std::unique_ptr<int16_t[]> txBuffer;

    explicit SDR(const SDRcfg::SDRConfig& cfg);
    virtual void initialize() = 0;
    virtual void sendSamples() = 0;
    virtual void receiveSamples() = 0;
    virtual ~SDR() = default;

   protected:
    void allocateBuffers();
};

class SDRConfigManager {
   private:
    std::vector<SDRcfg::SDRConfig> sdrConfigs;

   public:
    void loadFromNode(const fkyaml::node& sdrNode);
    const std::vector<SDRcfg::SDRConfig>& getConfigs() const;
};

struct SystemConfig {
    std::string logLevel;
    std::string logFile;
    size_t maxThreads;

    SystemConfig();

    void loadFromNode(const fkyaml::node& systemNode);
    void validate() const;
};

namespace SDRcfg {

enum SDRDeviceType { SoapySDR, UHD, Custom, UnknownType };
enum GainMode { Manual, SlowAttack, FastAttack, UnknownGain };
enum DataSourceType { File, Network, UnknowSource };

struct SDRConfig {
    SDRDeviceType deviceType;   // Тип устройства
    std::string name;           // Название устройства
    std::string deviceAddress;  // Адрес устройства

    double rxFrequency;   // Частота RX
    double rxSampleRate;  // Частота выборки RX
    double rxBandwidth;   // Полоса пропускания RX

    double txFrequency;   // Частота TX
    double txSampleRate;  // Частота выборки TX
    double txBandwidth;   // Полоса пропускания TX

    double gain;        // Усиление
    GainMode gainMode;  // Режим AGC

    size_t bufferSize;  // Размер буфера
    size_t multiplier;  // Множитель для сэмплов

    DataSourceType dataSourceType;  // Тип источника данных
    std::string dataSourcePath;  // Путь к источнику данных
    size_t repeatCount;  // Количество повторений (0 = бесконечно)

    SDRConfig(SDRDeviceType type, const std::string& name,
              const std::string& address, double rxFreq, double rxRate,
              double rxBW, double txFreq, double txRate, double txBW, double g,
              GainMode mode, size_t bufSize, size_t mult,
              DataSourceType srcType, const std::string& srcPath,
              size_t repeat);

    SDRConfig(const SDRConfig& other);
    SDRConfig();
};

}

ThreadManager::ThreadManager(size_t maxThreads)
    : running(true),
      maxThreads(maxThreads),
      groupRunning(std::bitset<MAX_THREAD_GROUP>().set()),
      tasks_in_queue(0) {
    threads.reserve(maxThreads);
    for (size_t i = 0; i < maxThreads; ++i) {
        threads.emplace_back(&ThreadManager::workerThread, this);
    }
}

ThreadManager::~ThreadManager() { stopAll(); }

void ThreadManager::addTask(const Task& task, TaskPriority priority,
                            size_t groupID) {
    if (groupID >= MAX_THREAD_GROUP) {
        std::cerr << "Invalid group ID: " << groupID << std::endl;
        return;
    }
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        taskQueue.push(TaskEntry{task, priority, groupID});
        tasks_in_queue++;
    }
    cv.notify_one();
}

void ThreadManager::stopAll() {
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        running.store(false);
    }
    cv.notify_all();

    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    threads.clear();
}

void ThreadManager::stopGroup(size_t groupID) {
    if (groupID >= MAX_THREAD_GROUP) {
        std::cerr << "Invalid group ID: " << groupID << std::endl;
        return;
    }

    std::bitset<MAX_THREAD_GROUP> expected = groupRunning.load();
    std::bitset<MAX_THREAD_GROUP> desired = expected;
    desired.reset(groupID);  // Сбрасываем нужный бит

    while (!groupRunning.compare_exchange_weak(expected, desired)) {
        desired = expected;
        desired.reset(groupID);
    }

    {
        std::lock_guard<std::mutex> lock(queueMutex);
        // std::priority_queue<TaskEntry> newQueue;
        std::priority_queue<TaskEntry, std::vector<TaskEntry>,
                            std::greater<TaskEntry>>
            newQueue;
        while (!taskQueue.empty()) {
            auto taskEntry = taskQueue.top();
            taskQueue.pop();
            if (taskEntry.groupID != groupID) {
                newQueue.push(taskEntry);
            }
        }
        taskQueue = std::move(newQueue);
    }
}

void ThreadManager::resizeThreadPool(size_t newSize) {
    if (newSize < maxThreads) {
        size_t threadsToStop = maxThreads - newSize;

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            running.store(false);
        }
        cv.notify_all();

        for (size_t i = 0; i < threadsToStop; ++i) {
            if (threads.back().joinable()) {
                threads.back().join();
            }
            threads.pop_back();
        }

        maxThreads = newSize;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            running.store(true);
        }
        cv.notify_all();
    } else {
        size_t threadsToAdd = newSize - maxThreads;
        for (size_t i = 0; i < threadsToAdd; ++i) {
            threads.emplace_back(&ThreadManager::workerThread, this);
        }
        maxThreads = newSize;
    }
}

void ThreadManager::workerThread() {
    while (running.load()) {
        TaskEntry taskEntry;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            cv.wait(lock,
                    [this]() { return !taskQueue.empty() || !running.load(); });

            if (!running.load() && taskQueue.empty()) {
                return;
            }

            taskEntry = taskQueue.top();
            taskQueue.pop();
            tasks_in_queue--;

            if (tasks_in_queue == 0) {
                cv.notify_one();
            }
        }

        if (!groupRunning.load()[taskEntry.groupID]) {
            continue;
        }

        try {
            taskEntry.task();
        } catch (const std::exception& e) {
            std::cerr << "Task error: " << e.what() << std::endl;
        }
    }
}

void ThreadManager::waitForAll() {
    std::unique_lock<std::mutex> lock(queueMutex);
    cv.wait(lock, [this]() { return tasks_in_queue == 0; });
}