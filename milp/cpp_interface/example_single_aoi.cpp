#include "MilpClient.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const std::string milp_dir = argc > 1 ? argv[1] : "../..";
    const std::string input_path = argc > 2 ? argv[2] : "";

    const std::string input_json = R"json(
{
  "aoi": {"row": 3, "col": 4},
  "staging_position": [150, -50],
  "commander_AOI": ["A_3_4"],
  "grid_weather": {"c0": 0.20, "c1": 0.15, "c2": 0.40, "c3": 0.55, "c4": 0.70},
  "platforms": [
    {"pid": "U1", "type": "UAV",  "pos": [150, -50], "sensors_mounted": ["EO", "SAR", "ESM"], "munitions": {"HF": 0, "RKT": 0, "GUN": 0}, "alt": 2.0, "lost": false},
    {"pid": "U2", "type": "UAV",  "pos": [150, -50], "sensors_mounted": ["EO", "SAR", "ESM"], "munitions": {"HF": 0, "RKT": 0, "GUN": 0}, "alt": 2.0, "lost": false},
    {"pid": "U3", "type": "UAV",  "pos": [150, -50], "sensors_mounted": ["EO", "SAR", "ESM"], "munitions": {"HF": 0, "RKT": 0, "GUN": 0}, "alt": 2.0, "lost": false},
    {"pid": "U4", "type": "UAV",  "pos": [150, -50], "sensors_mounted": ["EO", "SAR", "ESM"], "munitions": {"HF": 0, "RKT": 0, "GUN": 0}, "alt": 2.0, "lost": false},
    {"pid": "U5", "type": "UAV",  "pos": [150, -50], "sensors_mounted": ["EO", "SAR", "ESM"], "munitions": {"HF": 0, "RKT": 0, "GUN": 0}, "alt": 2.0, "lost": false},
    {"pid": "H1", "type": "HELI", "pos": [150, -50], "sensors_mounted": ["MMW", "EOIR"], "munitions": {"HF": 16, "RKT": 76, "GUN": 1200}, "alt": 3.0, "lost": false},
    {"pid": "H2", "type": "HELI", "pos": [150, -50], "sensors_mounted": ["MMW", "EOIR"], "munitions": {"HF": 16, "RKT": 76, "GUN": 1200}, "alt": 3.0, "lost": false}
  ],
  "targets": [
    {"tid": "g1", "type": "RADAR", "pos": [162, 112], "value": 0.97, "threat": 0.91, "confirmed": true, "alive": true}
  ]
}
)json";

    try {
        milp::SingleAoiMilpClient client(milp_dir);
        std::cout << (input_path.empty()
            ? client.solveJson(input_json)
            : client.solveFile(input_path)) << std::endl;
        return 0;
    } catch (const milp::MilpException& exc) {
        std::cerr << "MILP error (" << exc.status_code() << "): "
                  << exc.what() << std::endl;
        return 1;
    }
}
