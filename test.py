import os
import csv
import time
import torch
import torchaudio
from tqdm import tqdm
from util.util import compute_matrics

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.spectro_img import compute_visuals


def save_results_to_csv(results, filename="results.csv"):
    # Check if file exists. If not, create it and write headers
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "Sample Rate",
                    "SNR (mean)",
                    "SNR (std)",
                    "Base SNR (mean)",
                    "Base SNR (std)",
                    "LSD (mean)",
                    "LSD (std)",
                    "Base LSD (mean)",
                    "Base LSD (std)",
                    "LSD HF (mean)",
                    "LSD HF (std)",
                    "Base LSD HF (mean)",
                    "Base LSD HF (std)",
                    "LSD LF (mean)",
                    "LSD LF (std)",
                    "Base LSD LF (mean)",
                    "Base LSD LF (std)",
                    "RTF (mean)",
                    "RTF (std)",
                    "RTF Reciprocal (mean)",
                    "RTF Reciprocal (std)",
                ]
            )

        writer.writerow(results)


if __name__ == "__main__":

    # Initialize the setup
    opt = TrainOptions().parse()
    opt.isTrain = False
    visualizer = Visualizer(opt)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.get_train_dataloader()
    model = create_model(opt)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stride = opt.segment_length - opt.gen_overlap

    print(
        f"Input sample rate: {opt.lr_sampling_rate}, Output sample rate: {opt.hr_sampling_rate}"
    )

    results = []
    # Run for 5 epochs
    for epoch in range(3):
        # Define lists to accumulate results
        snr_list = []
        base_snr_list = []
        lsd_list = []
        base_lsd_list = []
        lsd_hf_list = []
        base_lsd_hf_list = []
        lsd_lf_list = []
        base_lsd_lf_list = []
        # RTF: Real Time Factor
        rtf_list = []

        # Process each file in the dataset
        with torch.no_grad():
            lr_audio = next(iter(dataset))["seg_audio"].to(device)
            print(model.flops(lr_audio[0]))

            print(f"Processing {len(dataset)} files")
            # Process each file in the dataset
            for i, data in enumerate(tqdm(dataset)):
                # For-loop for each segment in the audio file
                audio = []
                # data["seg_audio"] is tensor [segments, samples], where segments is the number of segments in the audio file
                lr_audio = data["seg_audio"].to(device)
                start_time = time.time()
                for segment_idx in range(lr_audio.shape[0]):
                    if lr_audio.shape[0] > 1:
                        # Get the output from the model
                        sr_spectro, sr_audio, lr_pha, norm_param, lr_spectro = (
                            model.inference(lr_audio[segment_idx, :].squeeze(0))
                        )
                    else:
                        sr_spectro, sr_audio, lr_pha, norm_param, lr_spectro = (
                            model.inference(lr_audio.squeeze(0))
                        )
                    audio.append(sr_audio)

                # Concatenate the audio
                if opt.gen_overlap > 0:
                    from torch.nn.functional import fold

                    out_len = (
                        data["seg_audio"].shape[0] - 1
                    ) * stride + opt.segment_length
                    audio = torch.cat(audio, dim=0)
                    audio[..., : opt.gen_overlap] *= 0.5
                    audio[..., -opt.gen_overlap :] *= 0.5
                    audio = audio.squeeze().transpose(-1, -2)
                    audio = fold(
                        audio,
                        kernel_size=(1, opt.segment_length),
                        stride=(1, stride),
                        output_size=(1, out_len),
                    ).squeeze(0)
                    audio = audio[..., opt.gen_overlap : -opt.gen_overlap]
                else:
                    audio = torch.cat(audio, dim=0).view(1, -1)

                run_time = time.time() - start_time
                rtf_list.append(
                    run_time
                    / ((lr_audio.shape[1] * lr_audio.shape[2]) / opt.sr_sampling_rate)
                )

                audio_len = int(data["auido_len"][0])
                # Evaluate the matrics
                (
                    snr,
                    base_snr,
                    lsd,
                    base_lsd,
                    lsd_hf,
                    base_lsd_hf,
                    lsd_lf,
                    base_lsd_lf,
                ) = compute_matrics(
                    data["raw_audio"].to(device),
                    data["lr_audio"][:, :audio_len].to(device),
                    audio[:, :audio_len].to(device),
                    opt,
                )

                # print(
                #     f"SNR: {snr}, Base SNR: {base_snr}, LSD: {lsd}, Base LSD: {base_lsd}, LSD HF: {lsd_hf}, Base LSD HF: {base_lsd_hf}, LSD LF: {lsd_lf}, Base LSD LF: {base_lsd_lf}"
                # )

                # Store metrics for each file
                snr_list.append(snr)
                base_snr_list.append(base_snr)
                lsd_list.append(lsd)
                base_lsd_list.append(base_lsd)
                lsd_hf_list.append(lsd_hf)
                base_lsd_hf_list.append(base_lsd_hf)
                lsd_lf_list.append(lsd_lf)
                base_lsd_lf_list.append(base_lsd_lf)

                if epoch == 0:
                    output_dir = (
                        f"./test_samples/{opt.hr_sampling_rate}/{opt.lr_sampling_rate}"
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    # Save the audio files
                    sr_path = os.path.join(
                        output_dir, f"{data['filename'][0][:-4]}_up.wav"
                    )
                    lr_path = os.path.join(
                        output_dir, f"{data['filename'][0][:-4]}_down.wav"
                    )
                    hr_path = os.path.join(
                        output_dir, f"{data['filename'][0][:-4]}_orig.wav"
                    )
                    # Save audio in 16-bit PCM format using torchaudio
                    torchaudio.save(
                        sr_path,
                        audio[:, :audio_len].cpu().detach().to(torch.float32),
                        opt.hr_sampling_rate,
                        bits_per_sample=16,
                    )
                    torchaudio.save(
                        lr_path,
                        data["lr_audio"],
                        opt.hr_sampling_rate,
                        bits_per_sample=16,
                    )
                    torchaudio.save(
                        hr_path,
                        data["raw_audio"],
                        opt.hr_sampling_rate,
                        bits_per_sample=16,
                    )

        # Compute the mean of the metrics
        snr = torch.stack(snr_list, dim=0).mean()
        base_snr = torch.stack(base_snr_list, dim=0).mean()
        lsd = torch.stack(lsd_list, dim=0).mean()
        base_lsd = torch.stack(base_lsd_list, dim=0).mean()
        lsd_hf = torch.stack(lsd_hf_list, dim=0).mean()
        base_lsd_hf = torch.stack(base_lsd_hf_list, dim=0).mean()
        lsd_lf = torch.stack(lsd_lf_list, dim=0).mean()
        base_lsd_lf = torch.stack(base_lsd_lf_list, dim=0).mean()
        rtf = torch.tensor(rtf_list).mean().to(device)
        rtf_reciprocal = 1 / rtf
        dict = {
            "snr": f"{snr.item():.2f}",
            "base_snr": f"{base_snr.item():.2f}",
            "lsd": f"{lsd.item():.2f}",
            "base_lsd": f"{base_lsd.item():.2f}",
            "lsd_hf": f"{lsd_hf.item():.2f}",
            "base_lsd_hf": f"{base_lsd_hf.item():.2f}",
            "lsd_lf": f"{lsd_lf.item():.2f}",
            "base_lsd_lf": f"{base_lsd_lf.item():.2f}",
            "rtf": f"{rtf.item():.2f}",
            "rtf_reciprocal": f"{rtf_reciprocal.item():.2f}",
        }
        results.append(
            torch.stack(
                [
                    snr,
                    base_snr,
                    lsd,
                    base_lsd,
                    lsd_hf,
                    base_lsd_hf,
                    lsd_lf,
                    base_lsd_lf,
                    rtf,
                    rtf_reciprocal,
                ],
                dim=0,
            ).unsqueeze(-1)
        )
        print(dict)

    # Save results to csv
    # Loop the results and calculate mean and std of results
    results = torch.cat(results, dim=1)
    # Get mean and std in [[mean, std], [mean, std], ...] format
    results = torch.stack([results.mean(dim=1), results.std(dim=1)], dim=1)
    # Convert to [mean, std, mean, std]
    results = results.flatten().tolist()
    # Add sample rate to the beginning of the list
    results.insert(0, opt.lr_sampling_rate)

    if opt.hr_sampling_rate == 48000:
        save_results_to_csv(results, "results_48kHz.csv")
    elif opt.hr_sampling_rate == 16000:
        save_results_to_csv(results, "results_16kHz.csv")
    else:
        save_results_to_csv(results)
