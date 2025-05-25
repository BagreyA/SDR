#include <chrono>
#include <cmath>

#include <atomic>
#include <bitset>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#ifndef MAX_THREAD_GROUP
#define MAX_THREAD_GROUP 100
#endif

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

class ThreadManager {
   public:
    using Task = std::function<void()>;

    enum class TaskPriority { High, Normal, Low };

    struct TaskEntry {
        Task task;
        TaskPriority priority;
        size_t groupID;

        bool operator>(const TaskEntry& other) const {
            return priority > other.priority;
        }
    };

    ThreadManager(size_t maxThreads = std::thread::hardware_concurrency());
    ~ThreadManager();

    void addTask(const Task& task, TaskPriority priority = TaskPriority::Normal,
                 size_t groupID = 0);
    void stopAll();
    void stopGroup(size_t groupID);
    void resizeThreadPool(size_t newSize);

    void waitForAll();

   private:
    void workerThread();

    std::atomic<bool> running;
    size_t maxThreads;

    std::vector<std::thread> threads;
    std::priority_queue<TaskEntry, std::vector<TaskEntry>,
                        std::greater<TaskEntry>>
        taskQueue;
    std::atomic<std::bitset<MAX_THREAD_GROUP>> groupRunning;

    std::mutex queueMutex;
    std::condition_variable cv;
    std::atomic<size_t> tasks_in_queue;
};

bool isPrime(int number) {
    if (number <= 1) return false;
    for (int i = 2; i <= std::sqrt(number); ++i) {
        if (number % i == 0) return false;
    }
    return true;
}

void heavyTask(size_t id, int limit) {
    int count = 0;
    for (int i = 2; i < limit; ++i) {
        if (isPrime(i)) ++count;
    }
    std::cout << "Task " << id << " finished, primes counted: " << limit
              << "\n";
}

int main() {
    const size_t numThreads = 1;
    const size_t numTasks = 100;
    const int computationLimit = 1000000000;

    ThreadManager threadManager(numThreads);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < numTasks; ++i) {
        threadManager.addTask(
            [i, computationLimit]() { heavyTask(i, computationLimit); },
            ThreadManager::TaskPriority::Low);
    }

    threadManager.waitForAll();
    threadManager.stopAll();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Total execution time: " << duration.count() << " seconds\n";

    return 0;
    return 0;
}
