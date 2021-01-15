CPU: Intel(R) Xeon(R) CPU E31230 (8 core @ 1.6 Gz) 
graphics card: nVidia GM107 [GeForce GTX 750 Ti]
The bench task is image classification cats vs. dogs (previous kaggle contest) with pretrained ResNet18 network.  
Inference time on CPU takes 370 seconds, inference time on GPU takes 39 seconds, rougly 10x speedup. C++ re-writing takes 26 seconds, about 10% improvement.  
