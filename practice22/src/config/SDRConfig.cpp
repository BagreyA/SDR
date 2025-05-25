#include <regex>
#include <stdexcept>
#include <string>

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

class Config {
   private:
    SystemConfig systemConfig;
    SDRConfigManager sdrConfigManager;

   public:
    void loadFromFile(const std::string& filepath);
    const SystemConfig& getSystemConfig() const;
    const std::vector<SDRcfg::SDRConfig>& getSDRConfigs() const;
};

static double parseValueWithMultiplier(const std::string& value) {
    static const std::regex regex(R"((\d+(\.\d+)?)\s*([kMG]?Hz|[kM]?SPS))");
    std::smatch match;

    if (std::regex_match(value, match, regex)) {
        double number = std::stod(match[1]);
        std::string unit = match[3];

        if (unit == "Hz" || unit.empty()) {
            return number;
        } else if (unit == "kHz" || unit == "k") {
            return number * 1e3;
        } else if (unit == "MHz" || unit == "M") {
            return number * 1e6;
        } else if (unit == "GHz" || unit == "G") {
            return number * 1e9;
        } else if (unit == "SPS") {
            return number;
        } else if (unit == "KSPS" || unit == "kSPS") {
            return number * 1e3;
        } else if (unit == "MSPS" || unit == "MS") {
            return number * 1e6;
        }

        throw std::invalid_argument("Unsupported unit: " + unit);
    }

    throw std::invalid_argument("Invalid value format: " + value);
}

static SDRcfg::GainMode parseGainMode(const std::string& mode) {
    if (mode == "manual") {
        return SDRcfg::GainMode::Manual;
    } else if (mode == "slow_attack") {
        return SDRcfg::GainMode::SlowAttack;
    } else if (mode == "fast_attack") {
        return SDRcfg::GainMode::FastAttack;
    }
    return SDRcfg::GainMode::UnknownGain;
}

static SDRcfg::DataSourceType parseDataSourceType(const std::string& type) {
    if (type == "file") {
        return SDRcfg::DataSourceType::File;
    } else if (type == "network") {
        return SDRcfg::DataSourceType::Network;
    }
    return SDRcfg::DataSourceType::UnknowSource;
}

void SDRConfigManager::loadFromNode(const fkyaml::node& sdrNode) {
    for (const auto& node : sdrNode) {
        SDRcfg::SDRConfig sdr;

        sdr.name = node["name"].is_null()
                       ? "Unknown_SDR"
                       : node["name"].get_value<std::string>();
        sdr.deviceAddress =
            node["device_address"].is_null()
                ? ""
                : node["device_address"].get_value<std::string>();
        sdr.bufferSize = node["buffer_size"].is_null()
                             ? 0
                             : node["buffer_size"].get_value<size_t>();
        sdr.multiplier = node["multiplier"].is_null()
                             ? 1
                             : node["multiplier"].get_value<size_t>();

        // Обработка типа устройства
        auto type = node["device_type"].is_null()
                        ? "UnknowSource"
                        : node["device_type"].get_value<std::string>();
        if (type == "SoapySDR") {
            sdr.deviceType = SDRcfg::SDRDeviceType::SoapySDR;
        } else if (type == "UHD") {
            sdr.deviceType = SDRcfg::SDRDeviceType::UHD;
        } else if (type == "Custom") {
            sdr.deviceType = SDRcfg::SDRDeviceType::Custom;
        } else {
            sdr.deviceType = SDRcfg::SDRDeviceType::UnknownType;
        }

        // Настройки усиления
        if (!node["settings"].is_null()) {
            auto settingsNode = node["settings"];

            if (!settingsNode["gain"].is_null()) {
                sdr.gain = settingsNode["gain"].get_value<double>();
            }
            if (!settingsNode["gain_mode"].is_null()) {
                sdr.gainMode = parseGainMode(
                    settingsNode["gain_mode"].get_value<std::string>());
            }

            if (!settingsNode["rx"].is_null()) {
                auto rxNode = settingsNode["rx"];
                if (!rxNode["frequency"].is_null()) {
                    sdr.rxFrequency = parseValueWithMultiplier(
                        rxNode["frequency"].get_value<std::string>());
                }
                if (!rxNode["sample_rate"].is_null()) {
                    sdr.rxSampleRate = parseValueWithMultiplier(
                        rxNode["sample_rate"].get_value<std::string>());
                }
                if (!rxNode["bandwidth"].is_null()) {
                    sdr.rxBandwidth = parseValueWithMultiplier(
                        rxNode["bandwidth"].get_value<std::string>());
                }
            }

            if (!settingsNode["tx"].is_null()) {
                auto txNode = settingsNode["tx"];
                if (!txNode["frequency"].is_null()) {
                    sdr.txFrequency = parseValueWithMultiplier(
                        txNode["frequency"].get_value<std::string>());
                }
                if (!txNode["sample_rate"].is_null()) {
                    sdr.txSampleRate = parseValueWithMultiplier(
                        txNode["sample_rate"].get_value<std::string>());
                }
                if (!txNode["bandwidth"].is_null()) {
                    sdr.txBandwidth = parseValueWithMultiplier(
                        txNode["bandwidth"].get_value<std::string>());
                }
            }
        }

        // Источник данных
        if (!node["data_source"].is_null()) {
            auto dataSourceNode = node["data_source"];

            if (!dataSourceNode["type"].is_null()) {
                sdr.dataSourceType = parseDataSourceType(
                    dataSourceNode["type"].get_value<std::string>());
            }
            if (!dataSourceNode["file_path"].is_null()) {
                sdr.dataSourcePath =
                    dataSourceNode["file_path"].get_value<std::string>();
            }
            if (!dataSourceNode["repeat_count"].is_null()) {
                if (dataSourceNode["repeat_count"].is_string()) {
                    std::string repeatCountValue =
                        dataSourceNode["repeat_count"].get_value<std::string>();
                    if (repeatCountValue == "inf") {
                        sdr.repeatCount =
                            static_cast<size_t>(-1);  // Бесконечно
                    } else {
                        throw std::invalid_argument(
                            "Invalid string value for repeat_count: " +
                            repeatCountValue);
                    }
                } else if (dataSourceNode["repeat_count"].is_integer()) {
                    size_t repeatCount =
                        dataSourceNode["repeat_count"].get_value<size_t>();
                    sdr.repeatCount = (repeatCount == 0)
                                          ? static_cast<size_t>(-1)
                                          : repeatCount;
                } else {
                    throw std::invalid_argument(
                        "repeat_count must be a number or 'inf'");
                }
            }
        }

        sdrConfigs.push_back(sdr);
    }
}

const std::vector<SDRcfg::SDRConfig>& SDRConfigManager::getConfigs() const {
    return sdrConfigs;
}
