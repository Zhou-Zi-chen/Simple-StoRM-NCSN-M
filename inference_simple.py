# simple_storm_inference.py
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
from typing import Optional, Union

from storm_model import StoRMModel


class SimpleStoRMInference:
    """
    简化的StoRM推理器
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 sr: int = 16000):
        
        self.device = device
        self.sr = sr
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # STFT参数
        self.n_fft = 510
        self.hop_length = 128
        self.win_length = 510
        
        print(f"推理器初始化完成 (设备: {device}, 采样率: {sr}Hz)")
    
    def _load_model(self, model_path: str) -> StoRMModel:
        """加载模型"""
        model = StoRMModel(base_channels=32).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'ema_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['ema_state_dict'])
            print("  使用EMA模型权重")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("  使用普通模型权重")
        
        return model
    
    def _preprocess_audio(self, waveform: torch.Tensor, input_sr: int) -> torch.Tensor:
        """预处理音频 - 简化版本"""
        # 简化：确保是2D [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [samples] -> [1, samples]
        elif waveform.dim() == 3:
            # [batch, channels, samples] -> [channels, samples]
            waveform = waveform.squeeze(0)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 重采样
        if input_sr != self.sr:
            waveform = torchaudio.functional.resample(
                waveform, 
                orig_freq=input_sr, 
                new_freq=self.sr
            )
        
        return waveform
    
    def _simple_stft(self, waveform: torch.Tensor):
        """简化的STFT"""
        # 确保是2D [batch, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        window = torch.hann_window(self.win_length)
        
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True
        )
        
        # 转换为模型输入格式 [batch, 2, freq, time]
        real = stft.real.unsqueeze(1)
        imag = stft.imag.unsqueeze(1)
        
        return torch.cat([real, imag], dim=1)
    
    def _simple_istft(self, complex_spec: torch.Tensor, target_length: Optional[int] = None):
        """简化的ISTFT"""
        from typing import Optional
        
        # 提取实部和虚部
        real = complex_spec[:, 0, :, :]
        imag = complex_spec[:, 1, :, :]
        
        # 创建复数
        stft_complex = torch.complex(real, imag)
        
        window = torch.hann_window(self.win_length)
        
        # 计算输出长度
        freq_bins = complex_spec.shape[2]
        calculated_length = (freq_bins - 1) * self.hop_length
        
        # 使用目标长度或计算长度
        output_length = target_length if target_length is not None else calculated_length
        
        waveform = torch.istft(
            stft_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=output_length
        )
        
        return waveform
    
    def enhance_audio(self, 
                     noisy_waveform: torch.Tensor,
                     input_sr: int = 16000,
                     denoise_only: bool = True,
                     num_steps: int = 10) -> torch.Tensor:
        """
        增强音频 - 最简单可靠的版本
        
        Args:
            noisy_waveform: 输入波形
            input_sr: 输入采样率
            denoise_only: 是否仅使用判别模型
            num_steps: 扩散步数（仅当denoise_only=False时有效）
        """
        print(f"\n开始增强音频...")
        
        with torch.no_grad():
            # 1. 预处理
            waveform = self._preprocess_audio(noisy_waveform, input_sr)
            original_length = waveform.shape[1]
            print(f"原始长度: {original_length}样本 ({original_length/self.sr:.3f}秒)")
            
            # 2. STFT
            stft = self._simple_stft(waveform)
            print(f"STFT形状: {stft.shape}")
            
            # 3. 调整尺寸为8的倍数
            B, C, F, T = stft.shape
            if F % 8 != 0 or T % 8 != 0:
                target_F = ((F + 7) // 8) * 8
                target_T = ((T + 7) // 8) * 8
                stft = torch.nn.functional.interpolate(
                    stft,
                    size=(target_F, target_T),
                    mode='bilinear',
                    align_corners=False
                )
                print(f"调整STFT: {stft.shape}")
            
            # 4. 模型增强
            stft = stft.to(self.device)
            
            if denoise_only:
                enhanced_stft = self.model.enhance(stft, denoise_only=True)
                print(f"使用判别模型增强")
            else:
                enhanced_stft = self.model.enhance(stft, num_steps=num_steps, denoise_only=False)
                print(f"使用扩散模型增强 (步数: {num_steps})")
            
            print(f"增强STFT: {enhanced_stft.shape}")
            
            # 5. 恢复原始STFT尺寸
            enhanced_stft = enhanced_stft.cpu()
            if enhanced_stft.shape[2:] != (F, T):
                enhanced_stft = torch.nn.functional.interpolate(
                    enhanced_stft,
                    size=(F, T),
                    mode='bilinear',
                    align_corners=False
                )
            
            # 6. ISTFT
            enhanced_waveform = self._simple_istft(enhanced_stft, original_length)
            print(f"增强波形: {enhanced_waveform.shape}")
            
            # 7. 确保正确长度
            current_length = enhanced_waveform.shape[1]
            if current_length != original_length:
                if current_length > original_length:
                    enhanced_waveform = enhanced_waveform[:, :original_length]
                else:
                    padding = torch.zeros(1, original_length - current_length)
                    enhanced_waveform = torch.cat([enhanced_waveform, padding], dim=1)
            
            print(f"最终波形: {enhanced_waveform.shape}")
            print(f"增强完成: {enhanced_waveform.shape[1]}样本 ({enhanced_waveform.shape[1]/self.sr:.3f}秒)")
            
            return enhanced_waveform.squeeze()
    
    def process_file(self, 
                    input_path: str, 
                    output_path: str,
                    denoise_only: bool = True,
                    num_steps: int = 10,
                    verbose: bool = True) -> bool:
        """
        处理单个音频文件
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"处理文件: {input_path}")
            print(f"{'='*60}")
        
        try:
            # 加载音频
            waveform, sr = torchaudio.load(input_path)
            if verbose:
                print(f"加载音频: {waveform.shape}, {sr}Hz")
            
            # 增强音频
            enhanced = self.enhance_audio(
                waveform, 
                input_sr=sr, 
                denoise_only=denoise_only,
                num_steps=num_steps
            )
            
            # 确保正确的维度
            if enhanced.dim() == 1:
                enhanced = enhanced.unsqueeze(0)
            
            # 创建输出目录
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 保存结果
            torchaudio.save(output_path, enhanced, self.sr)
            
            if verbose:
                # 验证
                if os.path.exists(output_path):
                    loaded, loaded_sr = torchaudio.load(output_path)
                    duration = loaded.shape[1] / loaded_sr
                    print(f"\n✅ 处理成功!")
                    print(f"  保存到: {output_path}")
                    print(f"  时长: {duration:.2f}秒")
                    print(f"  大小: {os.path.getsize(output_path) / 1024:.1f}KB")
            
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
                        denoise_only: bool = True,
                        num_steps: int = 10,
                        suffix: str = '_enhanced') -> dict:
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
        
        print(f"\n批量处理:")
        print(f"  输入目录: {input_dir}")
        print(f"  输出目录: {output_dir}")
        print(f"  找到 {len(audio_files)} 个音频文件")
        
        # 处理统计
        stats = {
            'total': len(audio_files),
            'success': 0,
            'failed': 0
        }
        
        # 处理每个文件
        for audio_file in tqdm(audio_files, desc="处理音频文件"):
            # 生成输出路径
            relative_path = audio_file.relative_to(input_dir)
            output_filename = f"{audio_file.stem}{suffix}{audio_file.suffix}"
            output_path = output_dir / relative_path.parent / output_filename
            
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 处理文件
            success = self.process_file(
                str(audio_file),
                str(output_path),
                denoise_only=denoise_only,
                num_steps=num_steps,
                verbose=False
            )
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        # 打印统计信息
        print(f"\n处理统计:")
        print(f"  总计: {stats['total']}")
        print(f"  成功: {stats['success']}")
        print(f"  失败: {stats['failed']}")
        
        return stats


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='简化的StoRM音频增强推理器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个文件（快速模式）
  python simple_storm_inference.py -i noisy.wav -o enhanced.wav
  
  # 处理单个文件（完整扩散模式）
  python simple_storm_inference.py -i noisy.wav -o enhanced.wav --denoise-only 0 --steps 30
  
  # 批量处理目录
  python simple_storm_inference.py -i noisy_dir/ -o enhanced_dir/ --batch
  
  # 指定模型文件
  python simple_storm_inference.py -i noisy.wav -o enhanced.wav -m checkpoints/final_model.pt
        """
    )
    
    # 必需参数
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='输入音频文件或目录')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='输出路径（文件或目录）')
    
    # 可选参数
    parser.add_argument('-m', '--model', type=str, 
                       default='checkpoints/best_model.pt',
                       help='模型文件路径 (默认: checkpoints/best_model.pt)')
    
    parser.add_argument('--denoise-only', type=int, default=1,
                       help='是否仅使用判别模型 (1=是, 0=否, 默认: 1)')
    
    parser.add_argument('--steps', type=int, default=10,
                       help='扩散步数（仅当denoise-only=0时有效, 默认: 10)')
    
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式（当输入为目录时）')
    
    parser.add_argument('--suffix', type=str, default='_enhanced',
                       help='输出文件后缀 (默认: _enhanced)')
    
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda/cpu, 默认: 自动选择)')
    
    parser.add_argument('--sr', type=int, default=16000,
                       help='目标采样率 (默认: 16000)')
    
    return parser.parse_args()


def main():
    """主函数 - 支持命令行"""
    args = parse_args()
    
    # 设置设备
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 检查模型文件是否存在
    if not Path(args.model).exists():
        print(f"❌ 模型文件不存在: {args.model}")
        print(f"  请确保已训练模型或提供正确的模型路径")
        print(f"  可用的模型文件:")
        for model_file in Path('checkpoints').glob('*.pt'):
            print(f"    - {model_file}")
        return
    
    # 创建推理器
    print(f"\n🚀 初始化StoRM推理器...")
    inference = SimpleStoRMInference(
        model_path=args.model,
        device=args.device,
        sr=args.sr
    )
    
    # 检查输入路径
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ 输入路径不存在: {args.input}")
        return
    
    # 处理单个文件
    if input_path.is_file():
        print(f"\n📄 处理单个文件")
        
        # 确保输出是文件
        output_path = Path(args.output)
        if output_path.is_dir():
            # 如果输出是目录，在里面创建同名文件
            output_filename = f"{input_path.stem}{args.suffix}{input_path.suffix}"
            output_path = output_path / output_filename
        
        success = inference.process_file(
            str(input_path),
            str(output_path),
            denoise_only=bool(args.denoise_only),
            num_steps=args.steps,
            verbose=True
        )
        
        if success:
            print(f"\n🎉 处理完成!")
            print(f"  输出文件: {output_path}")
        else:
            print(f"\n❌ 处理失败")
    
    # 处理目录
    elif input_path.is_dir():
        print(f"\n📁 处理目录")
        
        # 如果输出是文件，转换为目录
        output_path = Path(args.output)
        if output_path.suffix.lower() in ['.wav', '.mp3', '.flac'] and not args.batch:
            print(f"⚠️  警告: 输入是目录但输出指定为单个文件")
            print(f"  使用 --batch 参数进行批量处理")
            print(f"  或者将输出指定为目录")
            return
        
        # 确保输出是目录
        if output_path.suffix:
            output_path = output_path.parent / output_path.stem
        
        # 批量处理
        stats = inference.process_directory(
            input_dir=str(input_path),
            output_dir=str(output_path),
            denoise_only=bool(args.denoise_only),
            num_steps=args.steps,
            suffix=args.suffix
        )
        
        if stats['success'] > 0:
            print(f"\n🎉 批量处理完成!")
            print(f"  输出目录: {output_path}")
    
    else:
        print(f"❌ 无效的输入路径: {args.input}")


def quick_demo():
    """快速演示（没有参数时运行）"""
    print("🎵 StoRM音频增强推理器 - 简化版")
    print("="*50)
    
    # 检查是否有必要的文件
    test_audio = "p232_009.wav"
    default_model = "checkpoints/best_model.pt"
    
    if Path(test_audio).exists() and Path(default_model).exists():
        print(f"\n找到测试文件:")
        print(f"  音频文件: {test_audio}")
        print(f"  模型文件: {default_model}")
        
        choice = input("\n是否运行演示? (y/n): ")
        if choice.lower() == 'y':
            print(f"\n运行演示...")
            
            inference = SimpleStoRMInference(
                model_path=default_model,
                device='cpu'
            )
            
            output_file = "demo_enhanced.wav"
            success = inference.process_file(
                test_audio,
                output_file,
                denoise_only=True,
                verbose=True
            )
            
            if success:
                print(f"\n✅ 演示成功!")
                print(f"  输出文件: {output_file}")
    else:
        print(f"\n缺少测试文件:")
        if not Path(test_audio).exists():
            print(f"  ❌ 音频文件不存在: {test_audio}")
        if not Path(default_model).exists():
            print(f"  ❌ 模型文件不存在: {default_model}")
        
        print(f"\n请使用命令行参数:")
        print(f"  python simple_storm_inference.py -i <输入文件> -o <输出文件>")
        print(f"\n使用 --help 查看完整选项")


if __name__ == "__main__":
    # 如果没有命令行参数，显示演示
    if len(sys.argv) == 1:
        quick_demo()
    else:
        main()