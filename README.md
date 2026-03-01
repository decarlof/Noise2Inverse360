<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>2.5D Noise2Inverse for Denoising CT Data</title>
  <style>
    :root { color-scheme: light dark; }
    body {
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      line-height: 1.5;
      margin: 0;
      padding: 2rem 1.25rem;
      max-width: 980px;
    }
    h1, h2, h3 { line-height: 1.25; }
    h1 { margin-top: 0; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    pre {
      padding: 0.9rem 1rem;
      overflow: auto;
      border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
      border-radius: 10px;
      background: color-mix(in srgb, currentColor 6%, transparent);
    }
    code {
      padding: 0.1rem 0.35rem;
      border-radius: 6px;
      background: color-mix(in srgb, currentColor 8%, transparent);
      border: 1px solid color-mix(in srgb, currentColor 14%, transparent);
    }
    pre code { padding: 0; border: 0; background: transparent; }
    hr { border: 0; border-top: 1px solid color-mix(in srgb, currentColor 14%, transparent); margin: 2rem 0; }
    .note { margin: 0.75rem 0 1rem; }
  </style>
</head>
<body>

  <h1>2.5D Noise2Inverse for Denoising CT Data</h1>

  <h2>Overview</h2>
  <p>
    This project provides an implementation of the Noise2Inverse (N2I) framework for denoising CT data without requiring
    ground truth images. It implements the N2I framework following the 2.5D approach utilizing adjacent slices in the
    deep learning model. This project utilizes a simple U-Net model with leaky relu and group norm as the CNN model for
    denoising.
  </p>

  <h2>Assumptions</h2>
  <ul>
    <li>
      All output of this project (training results, inference results, trained models) are saved inside the directory of
      the reconstructions. For example:
      <ul>
        <li>
          User: John Smith
          <ul>
            <li>
              Sample 1 Directory:
              <ul>
                <li>
                  Provided by the User:
                  <ul>
                    <li>Full Reconstruction (Directory)</li>
                    <li>Sub-Reconstruction 1 (Directory)</li>
                    <li>Sub Reconstruction 2 (Directory)</li>
                  </ul>
                </li>
                <li>
                  Provided by N2I:
                  <ul>
                    <li><code>config.yaml</code></li>
                    <li>TrainOutput (Directory)</li>
                    <li>denoised_slices (Directory)</li>
                    <li>denoised_volume (Directory)</li>
                  </ul>
                </li>
              </ul>
            </li>
          </ul>
        </li>
      </ul>
    </li>

    <li>
      Data for training/inference is already created
      <ul>
        <li>This project does not contain the capacity to generate the necessary reconstructions for training/inference</li>
      </ul>
    </li>

    <li>
      Data for training/inference is saved as <code>.tiff</code> files
      <ul>
        <li>Can be either <code>.tif</code> or <code>.tiff</code></li>
      </ul>
    </li>

    <li>Model type/size is the same for each dataset</li>

    <li>
      We've found the U-Net model with no skip connections, leaky relu, and group norm to be a robust choice for different samples
    </li>

    <li>Model can be used for inference even when training is still running</li>
  </ul>

  <h2>Features</h2>
  <ul>
    <li>
      Automatic batch size optimization for A100/V100 GPUs
      <ul>
        <li>
          CT image size, GPU system memory, and model size all contribute to how much GPU RAM is used. Therefore, we've included
          a feature that automatically determines the maximum batch size to reduce the chance of an OOM error.
        </li>
      </ul>
    </li>
    <li>Support for 2.5D inference with PyTorch</li>
    <li>Flexible plug-and-play for different samples across different users</li>
  </ul>

  <h2>Installation Instructions</h2>
  <ul>
    <li>This project utilizes miniconda with a dedicated virtual environment</li>
  </ul>

  <pre><code>conda env create -f n2i_environment.yml</code></pre>

  <p class="note">Packages include:</p>
  <ul>
    <li>
      albumentations
      <ul>
        <li>Used for data augmentation during training</li>
      </ul>
    </li>
    <li>pytorch (2.4.0)</li>
    <li>cuda (11.8)</li>
    <li>tifffile</li>
    <li>tqdm</li>
    <li>matplotlib</li>
    <li>skimage</li>
  </ul>

  <h2>Project Structure</h2>
  <ul>
    <li>
      N2I
      <ul>
        <li>
          Python Files
          <ul>
            <li>data.py</li>
            <li>data_util.py</li>
            <li>denoise_slice.py</li>
            <li>denoise_volume.py</li>
            <li>eval.py</li>
            <li>loss.py</li>
            <li>main.py</li>
            <li>model.py</li>
            <li>tiffs.py</li>
            <li>utils.py</li>
          </ul>
        </li>

        <li>
          Bash Scripts
          <ul>
            <li>train.sh</li>
            <li>denoise_slice.sh</li>
            <li>denoise_volume.sh</li>
          </ul>
        </li>

        <li>
          Yaml Files
          <ul>
            <li>environment.yaml</li>
            <li>baseline_config.yaml</li>
          </ul>
        </li>
      </ul>
    </li>
  </ul>

  <h2>Getting Started</h2>

  <h3>Bash scripts</h3>
  <p>You will need to add the path to the virtual environment in the <code>.sh</code> bash scripts.</p>

  <p>
    Usage of this project includes training, denoising an individual slice, and denoising a full volume — each utilizing a bash script requiring a config file that specifies the location of the data.
  </p>

  <p class="note">Config file:</p>
  <ul>
    <li>
      This project contains a baseline config file to be used for training and inference. Simply copy the file into the directory of reconstructions and add:
      <ul>
        <li>the path to the directory, and</li>
        <li>the name of directories containing the full reconstruction, sub reconstruction 1, and sub reconstruction 2</li>
      </ul>
      into the appropriate spots of the config file.
    </li>
  </ul>

  <h3>Training</h3>
  <ul>
    <li>
      Training is done utilizing the <code>train.sh</code> bash script. This file first deactivates the base conda env and activates the env designed for this project. The script then calls the <code>main.py</code> file that runs distributed training utilizing 2 GPUs. The only item that needs to be specified is the location of the config file used for training/inference that contains the location of the data.
    </li>
    <li>
      Example usage:
      <pre><code>bash train.sh /path/to/config.yaml</code></pre>
    </li>
    <li>
      Training Logic:
      <ul>
        <li>Load in training parameters specified by config file</li>
        <li>Setup DDP training</li>
        <li>Create a directory for the training output</li>
        <li>Load in dataset for training</li>
        <li>Initialize model/optimizer</li>
        <li>Randomly select a patch size specified in the config file</li>
        <li>Brief warmup period using L1Loss before including LCL loss that helps the model focus on edges</li>
        <li>Each epoch prints the average loss</li>
        <li>Models are saved based on the lowest lcl and validation losses and highest edge value</li>
        <li>After warmup, training statistics are reset</li>
        <li>Predicted images are saved every 5 epochs to help visualize the denoising process</li>
      </ul>
    </li>
  </ul>

  <h3>Denoise Slice</h3>
  <ul>
    <li>
      Given APS produced volumes can be quite large, this project includes the option to denoise a single slice to quickly examine the denoised images. This can be done by using the <code>denoise_slice.sh</code> script. Like <code>train.sh</code>, you will need to specify the path to the config file and the slice to denoise. The output will then be saved as a <code>.tiff</code> file in the <code>denoised_slices</code> folder.
    </li>
    <li>
      Example usage:
      <pre><code>bash denoise_slice.sh /path/to/config.yaml 500</code></pre>
    </li>
    <li>
      Denoise Slice Logic:
      <ul>
        <li>Load in parameters specified by config file</li>
        <li>Create directory for denoised slices (not deleted/recreated after each use; slices accumulate)</li>
        <li>Load in pre-trained model</li>
        <li>
          Fetch the slice to denoise
          <ul>
            <li>Since the project follows the 2.5D approach, the previous and next slices are also fetched (hidden from the user)</li>
          </ul>
        </li>
        <li>The slice is patched using a sliding window approach similar to training</li>
        <li>Load in the mean/standard deviation used during training and normalize the image</li>
        <li>Denoise the image</li>
        <li>Rescale the image back to its original value</li>
        <li>Save as <code>.tiff</code></li>
      </ul>
    </li>
  </ul>

  <h3>Denoise Volume</h3>
  <ul>
    <li>
      Denoising the full volume involves running the <code>denoise_volume.sh</code> script. To help facilitate quick evaluation, this project contains the option to denoise a subset of slices (e.g., 500–600) specified on the command line. If no range is specified/left blank, it is assumed the full volume is to be denoised. The output will be saved in the <code>denoised_volume</code> folder. This folder is deleted and recreated each time.
    </li>
    <li>
      Example usage:
      <pre><code>bash denoise_volume.sh /path/to/config.yaml/ 500 600</code></pre>
    </li>
    <li>
      Denoised volume logic:
      <ul>
        <li>Load in parameters specified by config file</li>
        <li>Create directory for denoised volume (deleted and recreated for each run)</li>
        <li>Load in model</li>
        <li>
          Load in data
          <ul>
            <li>Assume <code>.tiff</code> files</li>
            <li>Data is normalized</li>
            <li>Data is patched using a sliding window approach to match the patch size seen during training</li>
          </ul>
        </li>
        <li>Calculate the optimal batch size</li>
        <li>Initialize empty array of size (#CT slices, image height, image width)</li>
        <li>Denoise volume in mini-batches and insert them into array</li>
        <li>Rescale denoised volume</li>
        <li>
          Save denoised volume as <code>.tiffs</code>
          <ul>
            <li>If a subset was selected, an offset is specified so that the save file names match the start of the subset</li>
          </ul>
        </li>
      </ul>
    </li>
  </ul>

  <h2>Contributing</h2>
  <p>
    Contributions are welcomed that either: improve the results and/or improve the flexibility of the workflow. A few areas for improvement include:
  </p>
  <ul>
    <li>
      Training by fine-tuning an old model
      <ul>
        <li>
          Each training run trains a model from scratch. While effective, this can be time consuming. For users who create multiple reconstructions from a single sample or similar sample, fine-tuning a model from a previous acquisition provides a tremendous speedup.
        </li>
        <li>We've seen reductions from 8–12 hours (train from scratch) down to 30–60 minutes (fine-tuning).</li>
      </ul>
    </li>
    <li>
      Different models
      <ul>
        <li>Current project utilizes the U-Net which works well, but other/newer architectures may provide some improvements.</li>
      </ul>
    </li>
  </ul>

</body>
</html>
