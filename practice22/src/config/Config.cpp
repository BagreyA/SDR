#include <fstream>
#include <stdexcept>
#include <vector>

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

class Config {
   private:
    SystemConfig systemConfig;
    SDRConfigManager sdrConfigManager;

   public:
    void loadFromFile(const std::string& filepath);
    const SystemConfig& getSystemConfig() const;
    const std::vector<SDRcfg::SDRConfig>& getSDRConfigs() const;
};

void Config::loadFromFile(const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open config file: " + filepath);
    }

    fkyaml::node root = fkyaml::node::deserialize(ifs);

    // Парсинг системной конфигурации
    if (!root["system"].is_null() && !root["system"].empty()) {
        systemConfig.loadFromNode(root["system"]);
        systemConfig.validate();
    } else {
        throw std::runtime_error(
            "System configuration is missing in YAML file.");
    }

    // Парсинг конфигурации SDR
    if (!root["sdr"].is_null() && !root["sdr"].empty()) {
        sdrConfigManager.loadFromNode(root["sdr"]);
    } else {
        throw std::runtime_error("SDR configuration is missing in YAML file.");
    }
}

const SystemConfig& Config::getSystemConfig() const { return systemConfig; }

const std::vector<SDRcfg::SDRConfig>& Config::getSDRConfigs() const {
    return sdrConfigManager.getConfigs();
}
