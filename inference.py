# storm_inference_fixed.py
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Optional, List, Union
import warnings
import sys
warnings.filterwarnings('ignore')

from storm_model import StoRMModel


class EnhancedStoRMInference:
    """
    增强的StoRM推理器
    支持命令行参数、批量处理和质量优化
    """
    
    def __init__(self, 
                model_path: str,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                sr: int = 16000,
                n_fft: int = 510,
                hop_length: int = 128,
                num_steps: int = 30,
                use_ema: bool = True):
        """
        初始化推理器
        """
        self.device = device
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.num_steps = num_steps
        
        # 加载模型
        print(f"🔧 加载模型: {model_path}")
        self.model = self._load_model(model_path, use_ema)
        self.model.eval()
        
        # 创建汉宁窗
        self.window = torch.hann_window(self.win_length)
        
        print(f"✅ 推理器初始化完成:")
        print(f"   设备: {device}")
        print(f"   采样率: {sr} Hz")
        print(f"   STFT: n_fft={n_fft}, hop={hop_length}")
        print(f"   扩散步数: {num_steps}")
        print(f"   使用EMA: {use_ema}")
    
    def _load_model(self, model_path: str, use_ema: bool = True) -> StoRMModel:
        """加载模型检查点"""
        model = StoRMModel(base_channels=32).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if use_ema and 'ema_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['ema_state_dict'])
            print("   ✅ 使用EMA模型权重")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("   ✅ 使用普通模型权重")
        else:
            raise ValueError("检查点中没有找到有效的模型权重")
        
        return model
    
    def _preprocess_audio(self, 
                        waveform: torch.Tensor, 
                        input_sr: int) -> torch.Tensor:
        """
        预处理音频
        """
        # 确保是2D张量 [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 重采样到目标采样率
        if input_sr != self.sr:
            waveform = torchaudio.functional.resample(
                waveform, 
                orig_freq=input_sr, 
                new_freq=self.sr
            )
        
        return waveform
    
    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """STFT转换"""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            return_complex=True
        )
        
        real = stft.real.unsqueeze(1)
        imag = stft.imag.unsqueeze(1)
        
        return torch.cat([real, imag], dim=1)
    
    def _istft(self, 
                complex_spec: torch.Tensor, 
                target_length: Optional[int] = None) -> torch.Tensor:
        """ISTFT转换"""
        real = complex_spec[:, 0, :, :]
        imag = complex_spec[:, 1, :, :]
        
        stft_complex = torch.complex(real, imag)
        
        freq_bins = complex_spec.shape[2]
        calculated_length = (freq_bins - 1) * self.hop_length
        
        output_length = target_length or calculated_length
        
        waveform = torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(complex_spec.device),
            length=output_length
        )
        
        return waveform
    
    def _adjust_to_multiple_of_8(self, tensor: torch.Tensor) -> torch.Tensor:
        """调整张量尺寸为8的倍数"""
        B, C, F, T = tensor.shape
        
        target_F = ((F + 7) // 8) * 8
        target_T = ((T + 7) // 8) * 8
        
        if F != target_F or T != target_T:
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(target_F, target_T),
                mode='bilinear',
                align_corners=False
            )
        
        return tensor
    
    def enhance(self, 
                noisy_waveform: torch.Tensor,
                input_sr: int = 16000,
                mode: str = 'quality',
                progress_callback = None) -> torch.Tensor:
        """
        增强音频
        """
        # 设置增强参数
        if mode == 'fast':
            denoise_only = True
            num_steps = 10
        elif mode == 'balanced':
            denoise_only = False
            num_steps = self.num_steps // 2
        else:  # 'quality'
            denoise_only = False
            num_steps = self.num_steps
        
        print(f"\n🎯 增强模式: {mode}")
        print(f"   仅判别模型: {denoise_only}")
        print(f"   扩散步数: {num_steps}")
        
        with torch.no_grad():
            # 1. 预处理
            if progress_callback:
                progress_callback(0.1, "预处理音频...")
            
            waveform = self._preprocess_audio(noisy_waveform, input_sr)
            original_length = waveform.shape[1]
            
            print(f"📊 音频信息:")
            print(f"   原始长度: {original_length}样本 ({original_length/self.sr:.2f}秒)")
            
            # 2. STFT
            if progress_callback:
                progress_callback(0.2, "STFT转换...")
            
            stft = self._stft(waveform)
            print(f"   STFT形状: {stft.shape}")
            
            # 3. 调整尺寸为8的倍数
            if progress_callback:
                progress_callback(0.3, "调整尺寸...")
            
            stft_adjusted = self._adjust_to_multiple_of_8(stft)
            if stft_adjusted.shape != stft.shape:
                print(f"   调整后STFT: {stft_adjusted.shape}")
            
            # 4. 模型增强
            if progress_callback:
                if denoise_only:
                    progress_callback(0.4, "判别模型增强...")
                else:
                    progress_callback(0.4, "扩散模型增强...")
            
            stft_adjusted = stft_adjusted.to(self.device)
            
            try:
                enhanced_stft = self.model.enhance(
                    stft_adjusted, 
                    num_steps=num_steps, 
                    denoise_only=denoise_only
                )
                print(f"   ✅ 增强成功")
                print(f"   增强STFT: {enhanced_stft.shape}")
                
            except Exception as e:
                print(f"   ❌ 模型增强失败: {e}")
                print("   ⚠️ 回退到仅判别模型...")
                enhanced_stft = self.model.enhance(stft_adjusted, denoise_only=True)
            
            # 5. 恢复原始STFT尺寸
            if progress_callback:
                progress_callback(0.8, "恢复尺寸...")
            
            enhanced_stft = enhanced_stft.cpu()
            if enhanced_stft.shape[2:] != stft.shape[2:]:
                enhanced_stft = torch.nn.functional.interpolate(
                    enhanced_stft,
                    size=stft.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # 6. ISTFT
            if progress_callback:
                progress_callback(0.9, "ISTFT转换...")
            
            enhanced_waveform = self._istft(enhanced_stft, original_length)
            
            # 7. 长度调整
            current_length = enhanced_waveform.shape[1]
            if current_length != original_length:
                if current_length > original_length:
                    enhanced_waveform = enhanced_waveform[:, :original_length]
                else:
                    padding = torch.zeros(1, original_length - current_length)
                    enhanced_waveform = torch.cat([enhanced_waveform, padding], dim=1)
            
            print(f"   ✅ 增强完成")
            print(f"   输出长度: {enhanced_waveform.shape[1]}样本 ({enhanced_waveform.shape[1]/self.sr:.2f}秒)")
            
            if progress_callback:
                progress_callback(1.0, "完成!")
            
            return enhanced_waveform.squeeze()
    
    def process_file(self, 
                    input_path: Union[str, Path],
                    output_path: Union[str, Path],
                    mode: str = 'balanced',
                    verbose: bool = True) -> bool:
        """
        处理单个音频文件
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🎵 处理文件: {input_path}")
            print(f"{'='*60}")
        
        try:
            # 加载音频
            waveform, sr = torchaudio.load(str(input_path))
            if verbose:
                print(f"📥 加载音频: {waveform.shape}, {sr}Hz")
            
            # 进度回调函数
            def progress_callback(progress, message):
                if verbose:
                    print(f"   [{progress*100:3.0f}%] {message}")
            
            # 增强音频
            enhanced = self.enhance(
                waveform, 
                input_sr=sr, 
                mode=mode,
                progress_callback=progress_callback if verbose else None
            )
            
            # 确保正确的维度
            if enhanced.dim() == 1:
                enhanced = enhanced.unsqueeze(0)
            
            # 创建输出目录
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存结果
            torchaudio.save(str(output_path), enhanced, self.sr)
            
            if verbose:
                # 验证
                if output_path.exists():
                    loaded, loaded_sr = torchaudio.load(str(output_path))
                    duration = loaded.shape[1] / loaded_sr
                    print(f"\n✅ 处理成功!")
                    print(f"   💾 保存到: {output_path}")
                    print(f"   ⏱️  时长: {duration:.2f}秒")
                    print(f"   📊 大小: {output_path.stat().st_size / 1024:.1f}KB")
            
            return True
            
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def process_directory(self,
                        input_dir: Union[str, Path],
                        output_dir: Union[str, Path],
                        file_ext: str = '.wav',
                        mode: str = 'balanced',
                        suffix: str = '_enhanced',
                        overwrite: bool = False) -> dict:
        """
        批量处理目录中的音频文件
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 查找音频文件
        audio_files = list(input_dir.glob(f'*{file_ext}'))
        if not audio_files:
            audio_files = list(input_dir.rglob(f'*{file_ext}'))
        
        print(f"\n📁 批量处理")
        print(f"   输入目录: {input_dir}")
        print(f"   输出目录: {output_dir}")
        print(f"   找到 {len(audio_files)} 个音频文件")
        print(f"   增强模式: {mode}")
        
        # 处理统计
        stats = {
            'total': len(audio_files),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'failed_files': []
        }
        
        # 处理每个文件
        for audio_file in tqdm(audio_files, desc="处理音频文件"):
            # 生成输出路径
            relative_path = audio_file.relative_to(input_dir)
            output_filename = f"{audio_file.stem}{suffix}{audio_file.suffix}"
            output_path = output_dir / relative_path.parent / output_filename
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 检查是否已存在
            if output_path.exists() and not overwrite:
                print(f"   ⏭️  跳过 (已存在): {audio_file.name}")
                stats['skipped'] += 1
                continue
            
            # 处理文件
            success = self.process_file(
                audio_file,
                output_path,
                mode=mode,
                verbose=False
            )
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
                stats['failed_files'].append(str(audio_file))
        
        # 打印统计信息
        print(f"\n📊 处理统计:")
        print(f"   总计: {stats['total']}")
        print(f"   成功: {stats['success']} ✅")
        print(f"   失败: {stats['failed']} ❌")
        print(f"   跳过: {stats['skipped']} ⏭️")
        
        if stats['failed_files']:
            print(f"\n❌ 失败的文件:")
            for file in stats['failed_files']:
                print(f"   - {file}")
        
        return stats


def parse_args():
    """解析命令行参数 - 修复版本"""
    parser = argparse.ArgumentParser(
        description='StoRM音频增强推理器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个文件
  python storm_inference.py --input noisy.wav --output enhanced.wav
  
  # 批量处理目录
  python storm_inference.py --input noisy_dir/ --output enhanced_dir/ --batch
  
  # 高质量模式
  python storm_inference.py --input noisy.wav --output enhanced.wav --mode quality --steps 50
  
  # 快速模式（仅判别模型）
  python storm_inference.py --input noisy.wav --output enhanced.wav --mode fast
  
  # 指定模型文件
  python storm_inference.py --input noisy.wav --output enhanced.wav --model checkpoints/final_model.pt
        """
    )
    
    # 输入输出
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入音频文件或目录')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='输出路径（文件或目录）')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理模式（当输入为目录时）')
    
    # 模型参数
    parser.add_argument('--model', type=str, 
                        default='checkpoints/best_model.pt',
                        help='模型检查点路径 (默认: checkpoints/best_model.pt)')
    parser.add_argument('--no-ema', action='store_true',
                        help='不使用EMA模型权重')
    
    # 处理参数
    parser.add_argument('--mode', type=str, default='balanced',
                        choices=['fast', 'balanced', 'quality'],
                        help='增强模式: fast(快速), balanced(平衡), quality(高质量) (默认: balanced)')
    parser.add_argument('--steps', type=int, default=30,
                        help='扩散步数 (默认: 30)')
    parser.add_argument('--suffix', type=str, default='_enhanced',
                        help='输出文件后缀 (默认: _enhanced)')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的输出文件')
    
    # 音频参数
    parser.add_argument('--sr', type=int, default=16000,
                        help='目标采样率 (默认: 16000)')
    
    # 设备
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='设备 (默认: 自动选择)')
    
    return parser.parse_args()


def main():
    # 使用 allow_abbrev=False 避免参数缩写问题
    parser = argparse.ArgumentParser(description='StoRM音频增强推理器', 
                                    allow_abbrev=False)
    
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    # 检查模型文件是否存在
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {args.model}")
        print(f"   请确保已训练模型或提供正确的模型路径")
        print(f"   可用的模型文件:")
        checkpoints_dir = Path('checkpoints')
        if checkpoints_dir.exists():
            for model_file in checkpoints_dir.rglob('*.pt'):
                print(f"    - {model_file}")
        else:
            print(f"    - checkpoints目录不存在")
        return
    
    # 创建推理器
    print(f"\n🚀 初始化StoRM推理器...")
    inference = EnhancedStoRMInference(
        model_path=str(model_path),
        device=args.device,
        sr=args.sr,
        num_steps=args.steps,
        use_ema=not args.no_ema
    )
    
    # 检查输入路径
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"❌ 输入路径不存在: {args.input}")
        return
    
    # 处理单个文件
    if input_path.is_file():
        print(f"\n📄 处理单个文件模式")
        success = inference.process_file(
            input_path,
            output_path,
            mode=args.mode,
            verbose=True
        )
        
        if success:
            print(f"\n🎉 单个文件处理完成!")
        else:
            print(f"\n❌ 处理失败")
    
    # 处理目录
    elif input_path.is_dir():
        print(f"\n📁 处理目录模式")
        
        # 如果输出是目录或指定了批量模式
        if output_path.suffix.lower() in ['.wav', '.mp3', '.flac'] and not args.batch:
            print(f"⚠️  警告: 输入是目录但输出指定为单个文件")
            print(f"   使用 --batch 参数进行批量处理")
            print(f"   或者将输出指定为目录")
            return
        
        # 确保输出是目录
        if output_path.suffix:
            output_path = output_path.parent / output_path.stem
        
        # 批量处理
        stats = inference.process_directory(
            input_dir=input_path,
            output_dir=output_path,
            mode=args.mode,
            suffix=args.suffix,
            overwrite=args.overwrite
        )
        
        if stats['success'] > 0:
            print(f"\n🎉 批量处理完成!")
            print(f"   输出目录: {output_path}")
    
    else:
        print(f"❌ 无效的输入路径: {args.input}")


def quick_test():
    """快速测试"""
    print("快速测试StoRM推理器...")
    
    # 检查必要的文件
    test_audio = Path('p232_009.wav')
    default_model = Path('checkpoints/best_model.pt')
    
    if test_audio.exists():
        print(f"✅ 测试音频文件存在: {test_audio}")
    else:
        print(f"❌ 测试音频文件不存在: {test_audio}")
    
    if default_model.exists():
        print(f"✅ 默认模型文件存在: {default_model}")
    else:
        print(f"❌ 默认模型文件不存在: {default_model}")
        # 查找其他模型文件
        checkpoints_dir = Path('checkpoints')
        if checkpoints_dir.exists():
            model_files = list(checkpoints_dir.rglob('*.pt'))
            if model_files:
                print(f"   找到其他模型文件:")
                for model_file in model_files[:3]:  # 只显示前3个
                    print(f"    - {model_file}")
                if len(model_files) > 3:
                    print(f"    - ... 还有{len(model_files)-3}个文件")
    
    # 简单的增强测试
    if test_audio.exists():
        # 尝试查找模型文件
        model_files = list(Path('checkpoints').rglob('*.pt'))
        if model_files:
            model_path = model_files[0]  # 使用第一个找到的模型
            print(f"\n使用模型: {model_path}")
            
            inference = EnhancedStoRMInference(
                model_path=str(model_path),
                device='cpu',
                num_steps=10
            )
            
            success = inference.process_file(
                test_audio,
                'test_enhanced.wav',
                mode='fast',
                verbose=True
            )
            
            if success:
                print(f"\n✅ 测试成功!")
                print(f"   输出文件: test_enhanced.wav")
            else:
                print(f"\n❌ 测试失败")
        else:
            print(f"\n❌ 没有找到模型文件")
    else:
        print(f"\n⚠️  缺少测试文件，跳过测试")


if __name__ == "__main__":
    # 如果没有命令行参数，运行测试
    if len(sys.argv) == 1:
        print("🚀 StoRM音频增强推理器")
        print("="*50)
        print("使用方法: python storm_inference.py --input <输入> --output <输出>")
        print("\n常用命令:")
        print("  python storm_inference.py -i noisy.wav -o enhanced.wav")
        print("  python storm_inference.py -i noisy.wav -o enhanced.wav --mode quality --steps 50")
        print("  python storm_inference.py -i noisy.wav -o enhanced.wav --model checkpoints/final_model.pt")
        print("\n运行快速测试...")
        quick_test()
    else:
        main()