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

SoapySDRDriver::SoapySDRDriver(const SDRcfg::SDRConfig& cfg) : SDR(cfg) {}

void SoapySDRDriver::initialize() {
    std::cout << "Initializing SoapySDR with rxsample rate: "
              << config.rxSampleRate
              << " Hz, rxfrequency: " << config.rxFrequency
              << " Hz, and gain: " << config.gain << " dB." << std::endl;
}

void SoapySDRDriver::sendSamples() {
    std::cout << "Sending samples to SoapySDR" << std::endl;
}

void SoapySDRDriver::receiveSamples() {
    std::cout << "Receiving samples from SoapySDR" << std::endl;
}
