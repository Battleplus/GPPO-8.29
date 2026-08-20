#include "MilpClient.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const std::string milp_dir = argc > 1 ? argv[1] : "../..";
    const std::string input_path = argc > 2 ? argv[2] : "";
    //默认御用场景
    const std::string input_json = R"json(
{
  "aois": [
    {"id": "A_3_4", "row": 3, "col": 4, "priority": 0.7, "target_prior": 0.55, "target_value": 0.85, "target_threat": 0.60},
    {"id": "A_5_6", "row": 5, "col": 6, "priority": 0.8, "target_prior": 0.70, "target_value": 0.97, "target_threat": 0.91}
  ],
  "platforms": {
    "UAV": {
      "count": 5,
      "pos": [150, -50],
      "sensors": ["EO", "SAR", "ESM"],
      "munitions": {"HF": 0, "RKT": 0, "GUN": 0}
    },
    "HELI": {
      "count": 2,
      "pos": [150, -50],
      "sensors": ["MMW", "EOIR"],
      "munitions": {"HF": 16, "RKT": 76, "GUN": 1200}
    }
  },
  "targets": [
    {"tid": "g1", "type": "RADAR", "pos": [162, 112], "value": 0.97, "threat": 0.91, "confirmed": true, "alive": true},
    {"tid": "g2", "type": "CP",    "pos": [265, 238], "value": 0.85, "threat": 0.60, "confirmed": true, "alive": true}
  ],
  "staging_position": [150, -50],
  "grid_weather": {"c0": 0.20, "c1": 0.15, "c2": 0.40, "c3": 0.55, "c4": 0.70},
  "aoi_route_state": null,
  "execution_feedback": null,
  "cycle_id": 0
}
)json";

    try {
        milp::MultiAoiMilpClient client(milp_dir);
        std::cout << (input_path.empty()
            ? client.runJson(input_json)
            : client.runFile(input_path)) << std::endl;
        return 0;
    } catch (const milp::MilpException& exc) {
        std::cerr << "MILP error (" << exc.status_code() << "): "
                  << exc.what() << std::endl;
        return 1;
    }
}
