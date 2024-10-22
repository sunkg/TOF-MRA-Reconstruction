# TOF-MRA-Reconstruction
This Pytorch code is for the work of “MIP-enhanced Uncertainty-aware Network for Fast 7T Time-of-Flight MRA Reconstruction” by Kaicong Sun, Caohui Duan, Xin Lou, and Dinggang Shen. The proposed uncertainty-aware MRI reconstruction model integrates evidential deep learning into deep unrolling framework. Instead of performing intensity (point) estimation, it assumes each pixel intensity follows Student’s t-distribution and estimates the distribution parameters. The principle of our model is not only applicable for TOF-MRA, but also for other MRI sequences or imaging modalities. To fit different applications and datasets, the estimated three-parameter distribution (γ, ν, α) could be relaxed to the original four-parameter one (γ, ν, α, β) of the Student’s t-distribution.


Abstract:
Time-of-flight (TOF) magnetic resonance angiography (MRA) is the dominant non-contrast MR imaging method for visualizing intracranial vascular system. The employment of 7T MRI for TOF-MRA is of great interest due to its outstanding spatial resolution and vessel-tissue contrast. However, high-resolution 7T TOF-MRA is undesirably slow. Besides, due to complicated and thin structures of brain vessels, reliability of reconstructed vessels is of great importance. In this work, we propose an uncertainty-aware reconstruction model for accelerated 7T TOF-MRA, which combines the merits of deep unrolling and evidential deep learning, such that our model not only provides promising MRI reconstruction, but also supports uncertainty quantification within a single inference. Moreover, we propose a maximum intensity projection (MIP) loss for TOF-MRA reconstruction to improve the quality of MIP images. In the experiments, we have evaluated our model on a relatively large in-house multi-coil 7T TOF-MRA dataset from different aspects, showing promising superiority of our model compared to state-of-the-art methods in terms of both TOF-MRA reconstruction and uncertainty quantification. 



To train:
python3 train.py  --logdir /path-to-save --lr 1e-4 --sparsity 0.125 --batch_size 1 --num_recurrent 9 --stages 4  --chans 32 --lambda0 1 --lambda1 0.1 --lambda2 10 --lambda3 0.1 --lambda4 1e-3 --mask Equispaced --CL_num 640 36 --shape 640 576 --trainpath /path-to-training-data --evalpath /path-to-validation-data

To inference:
python3 eval.py  --model_path /path-to-bestmodel  --save_path /path-to-save  --GT True --save_img True
