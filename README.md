# Adaptive-Military-Decision-Making-Training-Framework

# Adaptive Military Decision-Making Training Framework

Neuroadaptive simulation framework for military decision-making training leveraging IoT-derived cognitive and emotional feedback - MODSIM World 2025

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Conference](https://img.shields.io/badge/Conference-MODSIM%20World%202025-green.svg)](https://modsimworld.org/)

## Overview

This project implements a neuroadaptive, closed-loop simulation framework designed to enhance military training by integrating real-time physiological monitoring. The system uses wearable IoT devices (EEG, heart rate monitors, galvanic skin response sensors) to continuously assess cognitive load and emotional stress, dynamically adjusting scenario complexity to match each trainee's current state.

The framework addresses limitations in traditional simulation-based education (SBE) systems that present static scenarios without adapting to the trainee's mental state, providing a more effective preparation for dynamic operational environments.

## Key Features

- **Real-time Physiological Monitoring**: Integration of EEG, heart rate variability (HRV), galvanic skin response (GSR), temperature, and accelerometry data
- **Adaptive Training Environment**: Dynamic adjustment of scenario complexity, pacing, and environmental stimuli based on cognitive load and stress levels
- **Recovery Module**: Heart rate variability (RMSSD) tracking for emotional regulation assessment pre- and post-training
- **Statistical Analysis**: Comprehensive APA-format statistical reporting for training effectiveness evaluation
- **Multi-modal Data Integration**: Combines real EEG data with wearable sensor inputs for holistic state assessment

## Methodology

The system employs a four-stage adaptive cycle:
1. **Collect** multimodal physiological data
2. **Estimate** cognitive load and stress states using Random Forest classification
3. **Select** adaptive actions based on current state
4. **Adjust** training environment (tactical complexity, time pressure, communication noise)

### Data Sources
- **EEG Data**: OpenNeuro dataset (ds004362) with 109 subjects
- **Wearable Data**: Wearable Exam Stress Dataset for physiological indicators
- **Features**: 16-dimensional feature set (11 EEG-derived + 5 wearable sensor features)

## Performance Metrics

The framework evaluation includes:
- **Decision Accuracy**: Correctness of tactical choices
- **Task Completion Time**: Efficiency under varying cognitive loads
- **Physiological Recovery**: RMSSD measurements for stress resilience
- **Learning Retention**: Skill acquisition and recall over repeated scenarios
- **Cognitive Failure Rates**: Error detection during high-stress periods

## Results Summary

Results from 110 iterations comparing adaptive vs. static training modes:

| Metric | Adaptive Mode | Static Mode | Significance |
|--------|---------------|-------------|--------------|
| Decision Accuracy | M=0.800 (SD=0.115) | M=0.805 (SD=0.116) | p=.701 (ns) |
| Task Time | M=3.47s (SD=0.852) | M=3.49s (SD=0.897) | p=.901 (ns) |
| Learning Retention | 2.05% improvement | 0.18% improvement | Adaptive advantage |
| RMSSD Recovery | 14.34→15.77 | 13.05→14.36 | Both significant |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/adaptive-military-training-framework.git
cd adaptive-military-training-framework

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Run the adaptive training simulation
python Anamaria.py
```

The simulation will:
1. Load random EEG data from the preprocessed dataset
2. Initialize both adaptive and static training modes
3. Run real-time training simulations with physiological feedback
4. Generate comprehensive statistical analysis reports
5. Display interactive VR training interface

### Key Components

- **StatisticalAnalyzer**: APA-format statistical analysis and reporting
- **AdaptiveMilitarySystem**: Core adaptive logic and action selection
- **MilitaryVRInterface**: Interactive training environment with pygame
- **LearningTracker**: Performance tracking and skill assessment

## Requirements

- Python 3.8+
- NumPy, Pandas, SciPy
- MNE (neurophysiological data processing)
- NeuroKit2 (physiological signal processing)
- Scikit-learn (machine learning)
- Pygame (interactive interface)
- Matplotlib, Seaborn (visualization)

## Dataset Requirements

The system expects:
- EEG data directory: `/path/to/preprocessed_all_subjects/`
- Wearable data directory: `/path/to/wearable_data/`

Modify the paths in the code to match your dataset locations.

## Research Application

This framework was developed for the MODSIM World 2025 conference paper titled "Adaptive Simulation-Based Training for Military Decision-Making: Leveraging IoT-Derived Cognitive and Emotional Feedback." The research demonstrates enhanced physiological recovery and learning retention through neuroadaptive training.

### Research Team
- **Anamaria Acevedo Diaz** - University of Central Florida
- **Ancuta Margondai** - University of Central Florida  
- **Dr. Mustapha Mouloua** - University of Central Florida
- Additional research team members from UCF

## Contributing

This research project welcomes contributions in the areas of:
- Enhanced physiological signal processing
- Advanced adaptive algorithms
- Additional sensor modality integration
- Validation studies with military personnel

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{acevedodiaz2025adaptive,
    title={Adaptive Simulation-Based Training for Military Decision-Making: Leveraging IoT-Derived Cognitive and Emotional Feedback},
    author={Acevedo Diaz, Anamaria and Margondai, Ancuta and Von Ahlefeldt, Cindy and Willox, Sara and Ezcurra, Valentina and Hani, Soraya and Antanavicius, Emma and Islam, Nikita and Mouloua, Mustapha},
    booktitle={MODSIM World 2025},
    year={2025},
    pages={1--10},
    publisher={MODSIM World}
}
```

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dr. Mustapha Mouloua for research guidance and support
- University of Central Florida Human Factors and Cognitive Psychology Lab
- MODSIM World 2025 conference organizers
- OpenNeuro and PhysioNet for providing open-access datasets
