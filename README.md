# Supplementary Material: Inference Demo for FGC-Net

This repository provides the official inference code, pre-trained weights, and a running demo for the manuscript: 
**"Towards Rule-Compliant Driving Perception: A Fine-Grained Collaborative Network with Spatial-Semantic Decoupling"** submitted to IEEE T-ITS.

To protect the core intellectual property prior to formal acceptance, this supplementary material acts as a "half-open source" release. It includes the complete model definition (`model.py`), the inference script (`tools/demo.py`), and the pre-trained weights to demonstrate real-time, fine-grained panoptic perception (vehicle detection, drivable area segmentation, and lane line detection). The full training pipeline and dataset building scripts will be made publicly available upon acceptance.

## 1. Environment Setup

The model is built upon PyTorch and VMamba. To ensure a smooth evaluation, please set up the environment using the provided `requirements.txt`.

**Recommended Specifications:**
* OS: Ubuntu 22.04 LTS (or compatible Linux distribution)
* Python: 3.10.x
* CUDA: 12.8 (Compute Capability >= 7.0)
* PyTorch: 2.8.0

**Installation Steps:**
1. Create a virtual environment:
   ```bash
   conda create -n fgcnet python=3.10 -y
   conda activate fgcnet
   
2. Install the dependencies:
   
    Note: The environment includes mamba-ssm==2.2.5 and causal-conv1d==1.5.2 which require a proper CUDA compilation environment. Ensure your nvcc compiler matches the PyTorch CUDA version (12.8) to prevent building errors.

   ```bash
   pip install -r requirements.txt
   
3. Running the Inference Demo:

   We provide a plug-and-play demo script to evaluate the fine-grained topological reasoning capabilities of FGC-Net under complex traffic scenarios.
   
   Execution Command:
   Simply run the following command from the root directory:
   ```bash
   python tools/demo.py

   Note: The script is designed with hardware adaptability. It will automatically utilize the GPU (cuda) if available for optimal performance, or fallback to cpu to ensure the inference can still be executed successfully on standard laptops.

Execution Command:
Simply run the following command from the root directory:
   
4. Expected Output
    Upon running tools/demo.py, the script will automatically load the pre-trained weights and process the sample images provided in the testing folder.

    What reviewers will observe:

    Terminal Logging: The console will output the model loading status and the actual inference latency (FPS) for each processed frame.

    Visual Results: The inferred images will be saved automatically. The visualizations effectively demonstrate FGC-Net's capability to:

   a. Accurately output rigid bounding boxes for vehicle detection.

   b. Classify drivable areas (DAs) explicitly into direct and indirect regions.

   c. Detect lane lines (LLs) and distinguish between solid (uncrossable) and dashed (crossable) physical boundaries.

   d. Maintain spatial consistency without semantic boundary overstepping, even in challenging long-tail scenarios (e.g., nighttime glare or unstructured intersections).

4. Contact & Full Release
    The comprehensive open-source repository, including training scripts, hyperparameter configurations, and the BDD100K-FG topological dataset annotations, is fully prepared and will be unsealed immediately upon the manuscript's acceptance.

    Thank you for your time and effort in reviewing our work.
   
## Acknowledgement

This repository incorporates the official implementation of **[VMamba](https://github.com/MzeroMiko/VMamba)** as our foundational backbone. We directly include their codebase to ensure a seamless and out-of-the-box evaluation experience for the reviewers. 

We express our sincere gratitude to the original authors for their pioneering work and for open-sourcing their code, which significantly accelerated our research.
