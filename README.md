# Simple-StoRM-NCSN-M
基于NCSN++M backbone的StoRM语音增强的简单实现，主体框架分为diffusion model&amp; predictive model，ncsnppm中基础模块置于base_components中实现

训练模型
python train_storm.py --data_root ./speech_data(your dataset root) --num_epochs 50
