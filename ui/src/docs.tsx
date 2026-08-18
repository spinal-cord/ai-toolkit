import React from 'react';
import { ConfigDoc } from '@/types';
import { IoFlaskSharp } from 'react-icons/io5';

const docs: { [key: string]: ConfigDoc } = {
  'config.name': {
    title: 'Training Name',
    description: (
      <>
        The name of the training job. This name will be used to identify the job in the system and will the the filename
        of the final model. It must be unique and can only contain alphanumeric characters, underscores, and dashes. No
        spaces or special characters are allowed.
      </>
    ),
  },
  gpuids: {
    title: 'GPU ID',
    description: (
      <>
        This is the GPU that will be used for training. Only one GPU can be used per job at a time via the UI currently.
        However, you can start multiple jobs in parallel, each using a different GPU.
      </>
    ),
  },
  'config.process[0].trigger_word': {
    title: 'Trigger Word',
    description: (
      <>
        Optional: This will be the word or token used to trigger your concept or character.
        <br />
        <br />
        When using a trigger word, If your captions do not contain the trigger word, it will be added automatically the
        beginning of the caption. If you do not have captions, the caption will become just the trigger word. If you
        want to have variable trigger words in your captions to put it in different spots, you can use the{' '}
        <code>{'[trigger]'}</code> placeholder in your captions. This will be automatically replaced with your trigger
        word.
        <br />
        <br />
        Trigger words will not automatically be added to your test prompts, so you will need to either add your trigger
        word manually or use the
        <code>{'[trigger]'}</code> placeholder in your test prompts as well.
      </>
    ),
  },
  'config.process[0].model.name_or_path': {
    title: 'Name or Path',
    description: (
      <>
        The name of a diffusers repo on Huggingface or the local path to the base model you want to train from. The
        folder needs to be in diffusers format for most models. For some models, such as SDXL and SD1, you can put the
        path to an all in one safetensors checkpoint here.
      </>
    ),
  },
  'datasets.control_path': {
    title: 'Control Dataset',
    description: (
      <>
        The control dataset needs to have files that match the filenames of your training dataset. They should be
        matching file pairs. These images are fed as control/input images during training. The control images will be
        resized to match the training images.
      </>
    ),
  },
  'datasets.multi_control_paths': {
    title: 'Multi Control Dataset',
    description: (
      <>
        The control dataset needs to have files that match the filenames of your training dataset. They should be
        matching file pairs. These images are fed as control/input images during training.
        <br />
        <br />
        For multi control datasets, the controls will all be applied in the order they are listed. If the model does not
        require the images to be the same aspect ratios, such as with Qwen/Qwen-Image-Edit-2509, then the control images
        do not need to match the aspect size or aspect ratio of the target image and they will be automatically resized
        to the ideal resolutions for the model / target images.
      </>
    ),
  },
  'datasets.num_frames': {
    title: 'Number of Frames',
    description: (
      <>
        This sets the number of frames to shrink videos to for a video dataset. If this dataset is images, set this to 1
        for one frame. If your dataset is only videos, frames will be extracted evenly spaced from the videos in the
        dataset.
        <br />
        <br />
        It is best to trim your videos to the proper length before training. Wan is 16 frames a second. Doing 81 frames
        will result in a 5 second video. So you would want all of your videos trimmed to around 5 seconds for best
        results.
        <br />
        <br />
        Example: Setting this to 81 and having 2 videos in your dataset, one is 2 seconds and one is 90 seconds long,
        will result in 81 evenly spaced frames for each video making the 2 second video appear slow and the 90second
        video appear very fast.
      </>
    ),
  },
  'datasets.do_i2v': {
    title: 'Do I2V',
    description: (
      <>
        For video models that support both I2V (Image to Video) and T2V (Text to Video), this option trains this
        dataset in I2V mode. The first frame is extracted and used as conditioning; the loss is computed only on
        generated frames (after the first).
        <br /><br />
        For <strong>image datasets</strong> (num_frames=1), enabling Do I2V still works: the image is used as
        conditioning AND the model predicts it as output — effectively behaving like T2V with an additional
        conditioning input. This allows video models trained in I2V mode to also learn image generation.
        For video datasets, at least 5 frames are recommended so the model learns motion beyond the first frame
        (the first latent frame is masked from loss).
        <br /><br />
        Can be combined with "Do T2V" to train both modes (each video element is used twice: once conditioned,
        once unconditional).
      </>
    ),
  },
  'datasets.do_t2v': {
    title: 'Do T2V',
    description: (
      <>
        For video models that support both I2V (Image to Video) and T2V (Text to Video), this option trains this
        dataset in pure T2V mode: no first-frame conditioning, all frames are generated freely from text.
        <br /><br />
        For <strong>image datasets</strong> (num_frames=1), this is the recommended option — it behaves
        like standard text-to-image training: no conditioning overhead, full gradient on the single frame.
        Works with both video and image diffusion backends.
        <br /><br />
        Can be combined with "Do I2V" to train both modes simultaneously (each video element is used twice:
        once conditioned, once unconditional), doubling the training steps per epoch for this dataset.
      </>
    ),
  },
  'datasets.caption_dropout_rate_t2v': {
    title: 'Caption Dropout Rate for T2V',
    description: (
      <>
        Caption dropout rate specifically for T2V (Text to Video) mode items in mixed I2V/T2V training.
        This allows you to control caption dropout independently for T2V vs I2V items.
        By default, this is 0 (no dropout for T2V), as T2V training typically requires captions.
        The main "Caption Dropout Rate" setting applies to I2V items when both modes are enabled.
      </>
    ),
  },
  'datasets.do_audio': {
    title: 'Do Audio',
    description: (
      <>
        For models that support audio with video, this option will load the audio from the video and resize it to match
        the video sequence. Since the video is automatically resized, the audio may drop or raise in pitch to match the
        new speed of the video. It is important to prep your dataset to have the proper length before training.
      </>
    ),
  },
  'datasets.audio_normalize': {
    title: 'Audio Normalize',
    description: (
      <>
        When loading audio, this will normalize the audio volume to the max peaks. Useful if your dataset has varying
        audio volumes. Warning, do not use if you have clips with full silence you want to keep, as it will raise the
        volume of those clips.
      </>
    ),
  },
  'datasets.audio_preserve_pitch': {
    title: 'Audio Preserve Pitch',
    description: (
      <>
        When loading audio to match the number of frames requested, this option will preserve the pitch of the audio if
        the length does not match training target. It is recommended to have a dataset that matches your target length,
        as this option can add sound distortions.
      </>
    ),
  },
  'datasets.flip': {
    title: 'Flip X and Flip Y',
    description: (
      <>
        You can augment your dataset on the fly by flipping the x (horizontal) and/or y (vertical) axis. Flipping a
        single axis will effectively double your dataset. It will result it training on normal images, and the flipped
        versions of the images. This can be very helpful, but keep in mind it can also be destructive. There is no
        reason to train people upside down, and flipping a face can confuse the model as a person's right side does not
        look identical to their left side. For text, obviously flipping text is not a good idea.
        <br />
        <br />
        Control images for a dataset will also be flipped to match the images, so they will always match on the pixel
        level.
      </>
    ),
  },
  'train.unload_text_encoder': {
    title: 'Unload Text Encoder',
    description: (
      <>
        Unloading text encoder will cache the trigger word and the sample prompts and unload the text encoder from the
        GPU. Captions in for the dataset will be ignored
      </>
    ),
  },
  'train.cache_text_embeddings': {
    title: 'Cache Text Embeddings',
    description: (
      <>
        <small>(experimental)</small>
        <br />
        Caching text embeddings will process and cache all the text embeddings from the text encoder to the disk. The
        text encoder will be unloaded from the GPU. This does not work with things that dynamically change the prompt
        such as trigger words, caption dropout, etc.
      </>
    ),
  },
  'model.multistage': {
    title: 'Stages to Train',
    description: (
      <>
        Some models have multi stage networks that are trained and used separately in the denoising process. Most
        common, is to have 2 stages. One for high noise and one for low noise. You can choose to train both stages at
        once or train them separately. If trained at the same time, The trainer will alternate between training each
        model every so many steps and will output 2 different LoRAs. If you choose to train only one stage, the trainer
        will only train that stage and output a single LoRA.
      </>
    ),
  },
  'train.switch_boundary_every': {
    title: 'Switch Boundary Every',
    description: (
      <>
        When training a model with multiple stages, this setting controls how often the trainer will switch between
        training each stage.
        <br />
        <br />
        For low vram settings, the model not being trained will be unloaded from the gpu to save memory. This takes some
        time to do, so it is recommended to alternate less often when using low vram. A setting like 10 or 20 is
        recommended for low vram settings.
        <br />
        <br />
        The swap happens at the batch level, meaning it will swap between a gradient accumulation steps. To train both
        stages in a single step, set them to switch every 1 step and set gradient accumulation to 2.
      </>
    ),
  },
  'train.force_first_sample': {
    title: 'Force First Sample',
    description: (
      <>
        This option will force the trainer to generate samples when it starts. The trainer will normally only generate a
        first sample when nothing has been trained yet, but will not do a first sample when resuming from an existing
        checkpoint. This option forces a first sample every time the trainer is started. This can be useful if you have
        changed sample prompts and want to see the new prompts right away.
      </>
    ),
  },
  'model.layer_offloading': {
    title: (
      <>
        Layer Offloading{' '}
        <span className="text-yellow-500">
          ( <IoFlaskSharp className="inline text-yellow-500" name="Experimental" /> Experimental)
        </span>
      </>
    ),
    description: (
      <>
        This is an experimental feature based on{' '}
        <a className="text-blue-500" href="https://github.com/lodestone-rock/RamTorch" target="_blank">
          RamTorch
        </a>
        . This feature is early and will have many updates and changes, so be aware it may not work consistently from
        one update to the next. It will also only work with certain models.
        <br />
        <br />
        Layer Offloading uses the CPU RAM instead of the GPU ram to hold most of the model weights. This allows training
        a much larger model on a smaller GPU, assuming you have enough CPU RAM. This is slower than training on pure GPU
        RAM, but CPU RAM is cheaper and upgradeable. You will still need GPU RAM to hold the optimizer states and LoRA
        weights, so a larger card is usually still needed.
        <br />
        <br />
        You can also select the percentage of the layers to offload. It is generally best to offload as few as possible
        (close to 0%) for best performance, but you can offload more if you need the memory.
      </>
    ),
  },
  'model.qie.match_target_res': {
    title: 'Match Target Res',
    description: (
      <>
        This setting will make the control images match the resolution of the target image. The official inference
        example for Qwen-Image-Edit-2509 feeds the control image is at 1MP resolution, no matter what size you are
        generating. Doing this makes training at lower res difficult because 1MP control images are fed in despite how
        large your target image is. Match Target Res will match the resolution of your target to feed in the control
        images allowing you to use less VRAM when training with smaller resolutions. You can still use different aspect
        ratios, the image will just be resizes to match the amount of pixels in the target image.
      </>
    ),
  },
  'train.diff_output_preservation': {
    title: 'Differential Output Preservation',
    description: (
      <>
        Differential Output Preservation (DOP) is a technique to help preserve class of the trained concept during
        training. For this, you must have a trigger word set to differentiate your concept from its class. For instance,
        You may be training a woman named Alice. Your trigger word may be "Alice". The class is "woman", since Alice is
        a woman. We want to teach the model to remember what it knows about the class "woman" while teaching it what is
        different about Alice. During training, the trainer will make a prediction with your LoRA bypassed and your
        trigger word in the prompt replaced with the class word. Making "photo of Alice" become "photo of woman". This
        prediction is called the prior prediction. Each step, we will do the normal training step, but also do another
        step with this prior prediction and the class prompt in order to teach our LoRA to preserve the knowledge of the
        class. This should not only improve the performance of your trained concept, but also allow you to do things
        like "Alice standing next to a woman" and not make both of the people look like Alice.
      </>
    ),
  },
  'train.blank_prompt_preservation': {
    title: 'Blank Prompt Preservation',
    description: (
      <>
        Blank Prompt Preservation (BPP) is a technique to help preserve the current models knowledge when unprompted.
        This will not only help the model become more flexible, but will also help the quality of your concept during
        inference, especially when a model uses CFG (Classifier Free Guidance) on inference. At each step during
        training, a prior prediction is made with a blank prompt and with the LoRA disabled. This prediction is then
        used as a target on an additional training step with a blank prompt, to preserve the model's knowledge when no
        prompt is given. This helps the model to not overfit to the prompt and retain its generalization capabilities.
      </>
    ),
  },
  'train.do_differential_guidance': {
    title: 'Differential Guidance',
    description: (
      <>
        Differential Guidance will amplify the difference of the model prediction and the target during training to make
        a new target. Differential Guidance Scale will be the multiplier for the difference. This is still experimental,
        but in my tests, it makes the model train faster, and learns details better in every scenario I have tried with
        it.
        <br />
        <br />
        The idea is that normal training inches closer to the target but never actually gets there, because it is
        limited by the learning rate. With differential guidance, we amplify the difference for a new target beyond the
        actual target, this would make the model learn to hit or overshoot the target instead of falling short.
        <br />
        <br />
        <img src="/imgs/diff_guidance.png" alt="Differential Guidance Diagram" className="max-w-full mx-auto" />
      </>
    ),
  },
  'dataset.num_repeats': {
    title: 'Num Repeats',
    description: (
      <>
        Number of Repeats will allow you to repeate the items in a dataset multiple times. This is useful when you are
        using multiple datasets and want to balance the number of samples from each dataset. For instance, if you have a
        small dataset of 10 images and a large dataset of 100 images, you can set the small dataset to have 10 repeats
        to effectively make it 100 images, making the two datasets occour equally during training.
      </>
    ),
  },
  'train.audio_loss_multiplier': {
    title: 'Audio Loss Multiplier',
    description: (
      <>
        When training audio and video, sometimes the video loss is so great that it outweights the audio loss, causing
        the audio to become distorted. If you are noticing this happen, you can increase the audio loss multiplier to
        give more weight to the audio loss. You could try something like 2.0, 10.0 etc. Warning, setting this too high
        could overfit and damage the model.
      </>
    ),
  },
  'train.timestep_range_overrides': {
    title: 'Per-Timestep Range Loss Overrides',
    description: (
      <>
        <strong>Overview</strong>
        <br />
        Per-Timestep Range Loss Overrides allow you to dynamically adjust loss weights during training based on
        the current timestep. This gives you fine-grained control over what the model learns at different stages
        of the denoising process.
        <br />
        <br />
        <strong>How Ranges Work</strong>
        <br />
        Ranges are specified in <strong>absolute model timesteps (0-1000)</strong>. No scaling or mapping is applied —
        ranges are used exactly as written.
        <br />
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>Range is <code>[start, end)</code> — inclusive of start, exclusive of end</li>
          <li>Descending ranges (e.g., 1000-500) match timesteps from start down to end+1</li>
          <li>Ascending ranges (e.g., 0-500) also work — match timesteps from start up to end-1</li>
          <li>First matching range wins (order matters if ranges overlap)</li>
        </ul>
        <br />
        <strong>Dual-Expert Models (e.g., Wan 2.2 14B)</strong>
        <br />
        For dual-expert models, each expert operates in its own timestep range. Your overrides automatically
        apply only when the active expert's timestep falls within a specified range:
        <br />
        <br />
        <table className="w-full text-xs border-collapse border border-gray-600">
          <thead>
            <tr className="bg-gray-700">
              <th className="border border-gray-600 px-2 py-1 text-left">Model</th>
              <th className="border border-gray-600 px-2 py-1 text-left">High-Noise Expert</th>
              <th className="border border-gray-600 px-2 py-1 text-left">Low-Noise Expert</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-600 px-2 py-1">Wan 2.2 14B I2V</td>
              <td className="border border-gray-600 px-2 py-1">timesteps 901-1000</td>
              <td className="border border-gray-600 px-2 py-1">timesteps 0-900</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-2 py-1">Wan 2.2 14B T2V</td>
              <td className="border border-gray-600 px-2 py-1">timesteps 876-1000</td>
              <td className="border border-gray-600 px-2 py-1">timesteps 0-875</td>
            </tr>
          </tbody>
        </table>
        <br />
        <strong>Targeting Specific Experts</strong>
        <br />
        Because each expert only sees its own timesteps, you can target experts by choosing ranges within their
        operating window:
        <br />
        <br />
        <table className="w-full text-xs border-collapse border border-gray-600">
          <thead>
            <tr className="bg-gray-700">
              <th className="border border-gray-600 px-2 py-1 text-left">Range</th>
              <th className="border border-gray-600 px-2 py-1 text-left">Affects</th>
              <th className="border border-gray-600 px-2 py-1 text-left">Use Case</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-600 px-2 py-1"><code>1000-900</code></td>
              <td className="border border-gray-600 px-2 py-1">High-noise only</td>
              <td className="border border-gray-600 px-2 py-1">Structure/motion learning</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-2 py-1"><code>800-400</code></td>
              <td className="border border-gray-600 px-2 py-1">Low-noise only</td>
              <td className="border border-gray-600 px-2 py-1">Mid-range refinement</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-2 py-1"><code>100-0</code></td>
              <td className="border border-gray-600 px-2 py-1">Low-noise only</td>
              <td className="border border-gray-600 px-2 py-1">Final detail polish</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-2 py-1"><code>950-850</code></td>
              <td className="border border-gray-600 px-2 py-1">Both (crosses boundary)</td>
              <td className="border border-gray-600 px-2 py-1">Transition region</td>
            </tr>
          </tbody>
        </table>
        <br />
        <strong>Boundary Behavior</strong>
        <br />
        The boundary timestep itself belongs to the <strong>low-noise expert</strong> (t ≤ boundary).
        For example, at boundary=900:
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>Range <code>1000-900</code>: matches timesteps 901-1000 (high-noise)</li>
          <li>Range <code>900-800</code>: matches timesteps 801-900, <strong>including 900</strong> (low-noise)</li>
          <li>Timestep 900 → uses the <code>900-800</code> range, not <code>1000-900</code></li>
        </ul>
        <br />
        <strong>Precedence (What Takes Priority)</strong>
        <br />
        When determining which weight to use, the system checks in this order:
        <br />
        <ol className="list-decimal list-inside ml-4 space-y-1">
          <li><strong>Per-Range Override</strong> (if timestep matches AND field is explicitly set)</li>
          <li><strong>Per-Expert Config</strong> (e.g., spectral_low_weight_low)</li>
          <li><strong>Global Config</strong> (e.g., spectral_low_weight)</li>
        </ol>
        <br />
        <strong>Important:</strong> This is <strong>selective per field</strong>. If you set only <code>flow_weight</code>
        in a range, all other weights still fall through to per-expert/global config.
        <br />
        <br />
        <strong>Available Overrides Per Range</strong>
        <br />
        <table className="w-full text-xs border-collapse border border-gray-600">
          <thead>
            <tr className="bg-gray-700">
              <th className="border border-gray-600 px-2 py-1 text-left">Category</th>
              <th className="border border-gray-600 px-2 py-1 text-left">Fields</th>
              <th className="border border-gray-600 px-2 py-1 text-left">Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-600 px-2 py-1 font-medium">Loss Weights</td>
              <td className="border border-gray-600 px-2 py-1">
                <code>flow_weight</code>, <code>spectral_weight</code>, <code>mse_weight</code>
              </td>
              <td className="border border-gray-600 px-2 py-1">Overall weight for each loss component</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-2 py-1 font-medium">Frequency Weights</td>
              <td className="border border-gray-600 px-2 py-1">
                <code>spectral_low_weight</code>, <code>spectral_mid_weight</code>, <code>spectral_high_weight</code>
              </td>
              <td className="border border-gray-600 px-2 py-1">Balance between structure (low) and texture (high)</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-2 py-1 font-medium">Spectral Filters</td>
              <td className="border border-gray-600 px-2 py-1">
                <code>spectral_low_cutoff</code>, <code>spectral_high_cutoff</code>, <code>spectral_lcr_weight</code>, <code>spectral_temporal_scale</code>
              </td>
              <td className="border border-gray-600 px-2 py-1">Frequency band boundaries and temporal scaling</td>
            </tr>
          </tbody>
        </table>
        <br />
        <strong>Practical Examples</strong>
        <br />
        <br />
        <strong>Example 1: Structure first, then texture</strong>
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>Range <code>1000-700</code>: <code>spectral_low_weight=2.0</code>, <code>spectral_high_weight=0.5</code></li>
          <li>Range <code>600-0</code>: <code>spectral_low_weight=0.5</code>, <code>spectral_high_weight=3.0</code></li>
        </ul>
        <br />
        <strong>Example 2: Disable flow loss in low-noise regime</strong>
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>Range <code>800-0</code>: <code>flow_weight=0</code> (completely disables flow loss below timestep 800)</li>
        </ul>
        <br />
        <strong>Example 3: Fine-tune frequency cutoffs per range</strong>
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>Range <code>1000-600</code>: <code>spectral_low_cutoff=0.2</code> (wider low-freq band for structure)</li>
          <li>Range <code>500-0</code>: <code>spectral_low_cutoff=0.1</code>, <code>spectral_high_cutoff=0.6</code> (more high-freq focus)</li>
        </ul>
        <br />
        <strong>Interaction with Global Settings</strong>
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li><code>reverse_gate</code> is <strong>always global</strong> — per-range overrides use the same gate curve</li>
          <li><code>flow_max_timestep</code> is <strong>always global</strong> — affects the gate calculation</li>
          <li>Per-range <code>flow_weight</code> scales the gated flow loss (including when reverse_gate is on)</li>
          <li>Setting <code>flow_weight=0</code> in a range completely disables flow loss for that range</li>
        </ul>
        <br />
        <strong>Quick Tips</strong>
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>Leave fields empty (null) to use the default per-expert/global value</li>
          <li>Use adjacent non-overlapping ranges (e.g., 1000-900 and 900-0) for clean expert targeting</li>
          <li>Start with simple overrides — adjust one weight at a time</li>
          <li>Monitor training logs to see which ranges are being hit and their effects</li>
        </ul>
      </>
    ),
  },
  'train.spectral_flow_max_timestep': {
    title: 'Flow Max Timestep',
    description: (
      <>
        <strong>Overview</strong>
        <br />
        <br />
        This value sets the cutoff for the <strong>timestep gate</strong> that decides at which denoising
        timesteps the optical flow loss is actually applied. It is a smooth ramp, not a hard on/off
        switch.
        <br />
        <br />
        <strong>How the gate works</strong>
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>
            <strong>Normal gate</strong> (default): <code>gate(t) = 1 - t/max_timestep</code>. The flow loss
            is fully active at t=0 and ramps linearly to 0 at t=max_timestep. Items with t &gt;
            max_timestep contribute no flow loss.
          </li>
          <li>
            <strong>Reverse gate</strong>: <code>gate(t) = t/max_timestep</code>. The flow loss is 0 at t=0
            and ramps to full strength at t=max_timestep (useful to enforce motion consistency early /
            at high noise).
          </li>
        </ul>
        <br />
        <strong>Effect of the max timestep value</strong>
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>
            <strong>Low value</strong> (e.g. 300): flow is only enforced in a narrow low-noise window
            [0, 300]. Fewer steps contribute, but the ones that do have a strong, reliable x0
            prediction.
          </li>
          <li>
            <strong>High value</strong> (e.g. 1000): flow is enforced across the entire denoising range,
            so on average more of each batch contributes to the flow loss.
          </li>
        </ul>
        <br />
        The default (800) keeps flow mostly in the low-noise regime, where the reconstructed clean
        latent (x0) that the flow loss compares between frames is most accurate.
        <br />
        <br />
        <strong>How batch size and per-item timesteps interact</strong>
        <br />
        Each item in a batch is sampled at its <strong>own independent timestep</strong>, so it has its own
        gate value. The flow loss is a <strong>ratio over the active items</strong>:{' '}
        <code>sum(gate * loss) / sum(gate)</code>. Items whose gate is 0 drop out of both the numerator
        and the denominator, so they are effectively conditioned out.
        <br />
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>
            A batch where only 1 of 8 items is in the valid range produces the{' '}
            <strong>same per-item flow signal magnitude</strong> as a batch where all 8 are active (it is{' '}
            <strong>not</strong> diluted to 1/8). So <code>flow_weight</code> keeps a stable,
            batch-size-independent meaning.
          </li>
          <li>
            If <strong>every</strong> item in a batch is gated out (all t &gt; max_timestep), the flow loss for
            that step is exactly 0 and contributes no gradient; the spectral / MSE losses still train
            normally. With per-item timesteps this is rare, and its probability shrinks quickly as
            batch size grows.
          </li>
          <li>
            Smaller batches therefore have a slightly higher chance of a fully-gated (zero-flow) step,
            especially with a low max_timestep.
          </li>
        </ul>
        <br />
        <strong>Monitoring</strong>
        <br />
        Watch the <code>flow/gate_mean</code> log: it is the expected fraction of the batch that actually
        drives the flow objective. A low value means most of your timesteps sit above max_timestep and
        the flow loss is rarely active (consider raising max_timestep or <code>flow_weight</code>).
      </>
    ),
  },
  'train.attention_tanh_softcap_enabled': {
    title: 'Attention Tanh Softcapping',
    description: (
      <>
        <strong>Overview</strong>
        <br />
        Applies tanh softcapping to attention scores before softmax, inspired by Gemma2 and Grok-1.
        This technique prevents attention scores from becoming too extreme, improving training stability
        and generalization.
        <br />
        <br />
        <strong>How It Works</strong>
        <br />
        Transforms attention scores using: <code className="bg-gray-700 px-1 rounded">soft_cap * tanh(score / soft_cap)</code>
        <br />
        <br />
        <table className="w-full text-xs border-collapse border border-gray-600">
          <thead>
            <tr className="bg-gray-700">
              <th className="border border-gray-600 px-2 py-1 text-left">Score Range</th>
              <th className="border border-gray-600 px-2 py-1 text-left">Without Softcap</th>
              <th className="border border-gray-600 px-2 py-1 text-left">With Softcap (30)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-600 px-2 py-1">Normal (-10 to 10)</td>
              <td className="border border-gray-600 px-2 py-1">Passed through</td>
              <td className="border border-gray-600 px-2 py-1">Almost unchanged</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-2 py-1">Extreme (30+)</td>
              <td className="border border-gray-600 px-2 py-1">Dominates softmax</td>
              <td className="border border-gray-600 px-2 py-1">Capped at ~30</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-2 py-1">Very extreme (100+)</td>
              <td className="border border-gray-600 px-2 py-1">Near-100% attention</td>
              <td className="border border-gray-600 px-2 py-1">Capped at ~30</td>
            </tr>
          </tbody>
        </table>
        <br />
        <strong>Soft Cap Value</strong>
        <br />
        Controls how aggressively scores are capped:
        <br />
        <br />
        <table className="w-full text-xs border-collapse border border-gray-600">
          <thead>
            <tr className="bg-gray-700">
              <th className="border border-gray-600 px-2 py-1 text-left">Range</th>
              <th className="border border-gray-600 px-2 py-1 text-left">Effect</th>
              <th className="border border-gray-600 px-2 py-1 text-left">Use When</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-600 px-2 py-1"><code>10-20</code></td>
              <td className="border border-gray-600 px-2 py-1">Strong capping</td>
              <td className="border border-gray-600 px-2 py-1">Unstable training, very sharp attention</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-2 py-1"><code>20-30</code></td>
              <td className="border border-gray-600 px-2 py-1">Moderate capping</td>
              <td className="border border-gray-600 px-2 py-1">Default recommendation</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-2 py-1"><code>30-50</code></td>
              <td className="border border-gray-600 px-2 py-1">Gentle capping</td>
              <td className="border border-gray-600 px-2 py-1">Subtle stabilization only</td>
            </tr>
          </tbody>
        </table>
        <br />
        <strong>Benefits</strong>
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>Prevents attention from collapsing to single tokens</li>
          <li>Reduces gradient explosion from extreme attention scores</li>
          <li>Improves training stability, especially with large batch sizes</li>
          <li>Helps with long-context training where attention can become too focused</li>
        </ul>
        <br />
        <strong>Attention Mask Support</strong>
        <br />
        Softcapping now works correctly with attention masks. Masks are integrated into the
        flex_attention score_mod function, allowing both features to work together:
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>Padding masks: masked positions get -inf score (zero attention weight)</li>
          <li>Causal masks: future positions are masked before softcapping</li>
          <li>Custom masks: any SDPA-compatible mask format is supported</li>
        </ul>
        <br />
        <strong>Monitoring & Logging</strong>
        <br />
        Training logs automatically report softcapping statistics every 500 steps:
        <br />
        <br />
        <table className="w-full text-xs border-collapse border border-gray-600">
          <thead>
            <tr className="bg-gray-700">
              <th className="border border-gray-600 px-2 py-1 text-left">Metric</th>
              <th className="border border-gray-600 px-2 py-1 text-left">Meaning</th>
              <th className="border border-gray-600 px-2 py-1 text-left">Good Range</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-600 px-2 py-1"><code>Scores capped</code></td>
              <td className="border border-gray-600 px-2 py-1">% of attention scores modified by softcap</td>
              <td className="border border-gray-600 px-2 py-1">0-5% (low = working as safety net)</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-2 py-1"><code>Max reduction</code></td>
              <td className="border border-gray-600 px-2 py-1">How much extreme scores were reduced</td>
              <td className="border border-gray-600 px-2 py-1">0-20% (lower = less intervention needed)</td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-2 py-1"><code>LSE (sharpness)</code></td>
              <td className="border border-gray-600 px-2 py-1">Attention entropy (lower = softer/more diffuse)</td>
              <td className="border border-gray-600 px-2 py-1">Depends on seq len, monitor trends</td>
            </tr>
          </tbody>
        </table>
        <br />
        <strong>Interpreting the logs:</strong>
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li><strong>0% capped:</strong> Softcapping not needed (attention scores well-behaved)</li>
          <li><strong>1-5% capped:</strong> Normal - acting as safety net for extreme scores</li>
          <li><strong>{'>'}10% capped:</strong> Consider lowering soft_cap or investigating attention behavior</li>
          <li><strong>Falling LSE over time:</strong> Attention becoming softer (softcapping effect increasing)</li>
        </ul>
        <br />
        <strong>Performance Optimizations</strong>
        <br />
        The implementation includes several optimizations to minimize overhead:
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li><strong>Hardware-accelerated tanh:</strong> Uses <code className="text-green-400">tanh.approx.f32</code> PTX instruction (same as Gemma2/Grok-1)</li>
          <li><strong>BlockMask caching:</strong> Reuses pre-computed masks to avoid expensive vmap tracing</li>
          <li><strong>Block sparsity:</strong> Skips computation for fully-masked blocks (padding, causal)</li>
        </ul>
        <br />
        <strong>Requirements</strong>
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>Requires PyTorch 2.5+ with flex_attention support</li>
          <li>Falls back silently to standard attention if unavailable</li>
          <li>Currently integrated with Wan attention processors</li>
          <li>Logging overhead: negligible (~0.1% training time)</li>
        </ul>
        <br />
        <strong>Default: Enabled with soft_cap=30</strong>
        <br />
        This provides gentle stabilization without noticeably affecting model behavior.
      </>
    ),
  },
  'train.attention_f32_rope_enabled': {
    title: 'Attention F32 RoPE Acceleration',
    description: (
      <>
        <strong>Overview</strong>
        <br />
        Uses float32 instead of float64 for rotary position embedding (RoPE) computations,
        providing a significant speedup while maintaining numerical stability.
        <br />
        <br />
        <strong>How It Works</strong>
        <br />
        Rotary embeddings apply position-dependent rotations to query/key vectors. The computation involves:
        <br />
        <code className="bg-gray-700 px-1 rounded">x_rotated = complex(hidden_states) * freqs</code>
        <br />
        <br />
        <strong>Dtype Comparison</strong>
        <br />
        <table className="w-full text-xs border-collapse border border-gray-600">
          <thead>
            <tr className="bg-gray-700">
              <th className="border border-gray-600 px-2 py-1 text-left">Dtype</th>
              <th className="border border-gray-600 px-2 py-1 text-left">Speed</th>
              <th className="border border-gray-600 px-2 py-1 text-left">Precision</th>
              <th className="border border-gray-600 px-2 py-1 text-left">Used By</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-600 px-2 py-1"><code>float64</code></td>
              <td className="border border-gray-600 px-2 py-1">Slow</td>
              <td className="border border-gray-600 px-2 py-1">Maximum</td>
              <td className="border border-gray-600 px-2 py-1">Toolkit default (conservative)</td>
            </tr>
            <tr className="bg-blue-900/20">
              <td className="border border-gray-600 px-2 py-1"><code>float32</code></td>
              <td className="border border-gray-600 px-2 py-1">~20-40% faster</td>
              <td className="border border-gray-600 px-2 py-1">Excellent</td>
              <td className="border border-gray-600 px-2 py-1"><strong>Recommended</strong></td>
            </tr>
            <tr>
              <td className="border border-gray-600 px-2 py-1"><code>bf16/fp16</code></td>
              <td className="border border-gray-600 px-2 py-1">Fastest</td>
              <td className="border border-gray-600 px-2 py-1">Reduced</td>
              <td className="border border-gray-600 px-2 py-1">Diffusers (input dtype)</td>
            </tr>
          </tbody>
        </table>
        <br />
        <strong>Performance Impact</strong>
        <br />
        For Wan models with many transformer blocks, RoPE computation happens at every attention layer.
        Switching from F64 to F32 can reduce attention overhead by ~20-40%, which compounds across all blocks.
        <br />
        <br />
        <strong>When to Disable</strong>
        <br />
        <ul className="list-disc list-inside ml-4 space-y-1">
          <li>You need absolute maximum precision for research/comparison purposes</li>
          <li>You're debugging numerical instability issues (to rule out dtype as cause)</li>
        </ul>
        <br />
        <strong>Default: Enabled (float32)</strong>
        <br />
        Provides the best balance of speed and stability for training.
      </>
    ),
  },
  'datasets.auto_frame_count': {
    title: 'Auto Frame Count',
    description: (
      <>
        This will automatically determine the number of frames to use for each video in your dataset instead of relying
        on a fixed num_frames. This allows you to include videos of different lengths in the dataset, and each video
        will be processed without speeding up or slowing down. Be careful about adding long videos into your dataset, as
        they use up more VRAM. This currently will not work with a batch size greater than 1.
      </>
    ),
  },
  'model.model_kwargs.kv_cache': {
    title: 'KV Cache',
    description: (
      <>
        This will enable KV Cache for control images in a model that supports it. LoRAs trained with this on
        need to also be inferenced with it, and vice versa. This does not speed up or slow down training, but on inference,
        the control images only need to be processed once for the entire generation, vs being processed for every step.
        Which leads to a significant speedup on inference.
      </>
    ),
  },

  // --- LoRA Init docs ---
  'config.process[0].network.lora_a_init': {
    title: 'LoRA A Matrix Init',
    description: (
      <>
        Controls how the LoRA A (down projection) matrix is initialized. The A matrix projects the input to a lower-
        dimensional space.
        <br />
        <br />
        <strong>Available Methods:</strong>
        <ul className="list-disc pl-5 mt-2 space-y-1">
          <li>
            <strong>Gaussian Random</strong> — Random values from a normal distribution. Default std = 1/√rank. Best
            for most cases as it preserves the original layer's output statistics. Recommended for LoRA A.
          </li>
          <li>
            <strong>Kaiming Uniform</strong> — Uniform distribution derived from the number of input features. Good for
            ReLU-like activations. The default used by Microsoft's LoRA implementation.
          </li>
          <li>
            <strong>Kaiming Normal</strong> — Normal distribution variant of Kaiming initialization. Similar to Kaiming
            Uniform but with Gaussian distribution.
          </li>
          <li>
            <strong>Xavier Uniform</strong> — Uniform distribution based on input/output fan. Good for tanh/sigmoid
            activations.
          </li>
          <li>
            <strong>Xavier Normal</strong> — Normal distribution variant of Xavier initialization.
          </li>
          <li>
            <strong>Normal</strong> — Standard normal distribution with configurable std (default 0.01). Use for small
            random perturbations.
          </li>
          <li>
            <strong>Zeros</strong> — All zeros. Typically used for LoRA B (up projection) so the initial LoRA output is
            zero, preserving the original model output before training.
          </li>
          <li>
            <strong>Small Noise</strong> — Normal distribution with std=0.001. Minimal perturbation from zero.
          </li>
        </ul>
        <br />
        <strong>Recommendation:</strong> <strong>Gaussian Random</strong> (default) is generally the best choice for
        LoRA A as it helps preserve the pre-trained weights' statistics. Use <strong>Zeros</strong> for LoRA B so the
        LoRA starts as a no-op.
      </>
    ),
  },
  'config.process[0].network.lora_a_init_std': {
    title: 'LoRA A Init Std',
    description: (
      <>
        Standard deviation for the Gaussian Random initialization of the LoRA A matrix.
        <br />
        <br />
        When left empty, the default is <code>1/√rank</code> (e.g., for rank 64, std ≈ 0.125). Lower values produce
        smaller initial weights, while higher values produce larger initial weights. This only applies when the init
        method is set to <strong>Gaussian Random</strong>.
      </>
    ),
  },
  'config.process[0].network.lora_b_init': {
    title: 'LoRA B Matrix Init',
    description: (
      <>
        Controls how the LoRA B (up projection) matrix is initialized. The B matrix projects back to the original
        dimension.
        <br />
        <br />
        <strong>Recommendation:</strong> <strong>Zeros</strong> is the standard choice for LoRA B. This ensures the
        LoRA adds zero to the original layer output at the start of training, effectively making the LoRA a no-op
        until training begins. Changing this is rarely needed.
      </>
    ),
  },
  'config.process[0].network.lora_b_init_std': {
    title: 'LoRA B Init Std',
    description: (
      <>
        Standard deviation for the Gaussian Random initialization of the LoRA B matrix.
        <br />
        <br />
        When left empty, the default is <code>1/√rank</code>. This only applies when the init method is set to
        <strong>Gaussian Random</strong>. Note: using anything other than Zeros for LoRA B will cause the LoRA to add
        non-zero values to the original output from the start of training.
      </>
    ),
  },
  'config.process[0].network.high_noise_lora_a_init': {
    title: 'High Noise LoRA A Init',
    description: (
      <>
        Per-expert initialization for the <strong>High Noise</strong> transformer (transformer_1) in multistage models.
        The high noise transformer handles the initial denoising steps and benefits from different initialization to
        better learn the coarse denoising patterns.
        <br />
        <br />
        If left empty, falls back to the general <strong>LoRA A Matrix Init</strong> setting above.
      </>
    ),
  },
  'config.process[0].network.high_noise_lora_a_init_std': {
    title: 'High Noise LoRA A Init Std',
    description: (
      <>
        Standard deviation for the Gaussian Random initialization of the High Noise LoRA A matrix.
        <br />
        <br />
        When left empty, the default is <code>1/√rank</code>. This only appears when the init method is set to
        <strong>Gaussian Random</strong>.
      </>
    ),
  },
  'config.process[0].network.high_noise_lora_b_init': {
    title: 'High Noise LoRA B Init',
    description: (
      <>
        Per-expert initialization for the <strong>High Noise</strong> transformer (transformer_1) in multistage models.
        <br />
        <br />
        If left empty, falls back to the general <strong>LoRA B Matrix Init</strong> setting above.
      </>
    ),
  },
  'config.process[0].network.high_noise_lora_b_init_std': {
    title: 'High Noise LoRA B Init Std',
    description: (
      <>
        Standard deviation for the Gaussian Random initialization of the High Noise LoRA B matrix.
        <br />
        <br />
        When left empty, the default is <code>1/√rank</code>. This only appears when the init method is set to
        <strong>Gaussian Random</strong>.
      </>
    ),
  },
  'config.process[0].network.low_noise_lora_a_init': {
    title: 'Low Noise LoRA A Init',
    description: (
      <>
        Per-expert initialization for the <strong>Low Noise</strong> transformer (transformer_2) in multistage models.
        The low noise transformer handles the final refinement steps and may benefit from different initialization to
        better learn fine details.
        <br />
        <br />
        If left empty, falls back to the general <strong>LoRA A Matrix Init</strong> setting above.
      </>
    ),
  },
  'config.process[0].network.low_noise_lora_a_init_std': {
    title: 'Low Noise LoRA A Init Std',
    description: (
      <>
        Standard deviation for the Gaussian Random initialization of the Low Noise LoRA A matrix.
        <br />
        <br />
        When left empty, the default is <code>1/√rank</code>. This only appears when the init method is set to
        <strong>Gaussian Random</strong>.
      </>
    ),
  },
  'config.process[0].network.low_noise_lora_b_init': {
    title: 'Low Noise LoRA B Init',
    description: (
      <>
        Per-expert initialization for the <strong>Low Noise</strong> transformer (transformer_2) in multistage models.
        <br />
        <br />
        If left empty, falls back to the general <strong>LoRA B Matrix Init</strong> setting above.
      </>
    ),
  },
  'config.process[0].network.low_noise_lora_b_init_std': {
    title: 'Low Noise LoRA B Init Std',
    description: (
      <>
        Standard deviation for the Gaussian Random initialization of the Low Noise LoRA B matrix.
        <br />
        <br />
        When left empty, the default is <code>1/√rank</code>. This only appears when the init method is set to
        <strong>Gaussian Random</strong>.
      </>
    ),
  },

  // --- Wan 2.2 Tensor-Type-Specific LoRA Configuration docs ---
  'config.process[0].network.wan22_tensor_types': {
    title: 'Wan 2.2 Tensor-Type-Specific Ranks',
    description: (
      <>
        Fine-grained control over which tensor types in Wan 2.2 are trained and their individual LoRA ranks.
        <br />
        <br />
        <strong>Tensor Types and Their Roles:</strong>
        <ul className="list-disc pl-5 mt-2 space-y-1">
          <li>
            <strong>Self Attention (self_attn)</strong> — Internal attention within the transformer blocks. Handles
            temporal and spatial coherence. Max rank: 5120.
          </li>
          <li>
            <strong>Cross Attention (cross_attn)</strong> — Connects visual features to text prompts. Critical for
            following instructions. Max rank: 5120.
          </li>
          <li>
            <strong>Feed Forward (ffn)</strong> — Feature transformation layers. Adds representational power.
            Max rank: 5120.
          </li>
          <li>
            <strong>Text Embedding (text_embedding)</strong> — Converts text encodings to model space. Important for
            prompt adherence. Max rank: 4096.
          </li>
          <li>
            <strong>Time Embedding (time_embedding)</strong> — Encodes timestep information for the denoising process.
            Max rank: 256.
          </li>
          <li>
            <strong>Output Head (head)</strong> — Final projection layer. Usually less important for style.
            Max rank: 64.
          </li>
        </ul>
        <br />
        <strong>Configuration Options:</strong>
        <ul className="list-disc pl-5 mt-2 space-y-1">
          <li>
            <strong>Rank:</strong> Set the LoRA rank for each tensor type. Leave empty to use the global Linear Rank.
            Setting to 0 or null skips that tensor type entirely.
          </li>
          <li>
            <strong>Max Button:</strong> Quickly sets the rank to the maximum supported value for that tensor type.
          </li>
          <li>
            <strong>Full Checkbox:</strong> Use full weight training (not LoRA) for this tensor type. This is useful
            for small layers like the output head or time embedding where full training is more efficient.
          </li>
          <li>
            <strong>Default Button:</strong> Resets to using the global Linear Rank for that type.
          </li>
        </ul>
        <br />
        <strong>Training Scenarios:</strong>
        <ul className="list-disc pl-5 mt-2 space-y-1">
          <li>
            <em>Style LoRA:</em> Train self_attn + cross_attn with moderate rank (128-512), skip ffn.
          </li>
          <li>
            <em>Character LoRA:</em> Train all main layers (self_attn, cross_attn, ffn) with equal rank.
          </li>
          <li>
            <em>Prompt Enhancement:</em> Train text_embedding at full rank, skip most others.
          </li>
          <li>
            <em>Minimal:</em> Train only one tensor type (e.g., text_embedding at max rank) for specialized effects.
          </li>
        </ul>
        <br />
        <em>Note: The rank is automatically clamped to the maximum supported value for each tensor type based on
        the model's architecture (Wan 2.2 14B).</em>
      </>
    ),
  },

  // --- Optimizer docs ---
  optimizer: {
    title: 'Optimizer',
    description: (
      <>
        The optimizer controls how the model's weights are updated during training based on the computed gradients.
        <br />
        <br />
        <strong>Standard Optimizers (work with LR Schedulers):</strong>
        <ul className="list-disc pl-5 mt-2 space-y-1">
          <li>
            <strong>Adam / AdamW</strong> — The default choice. AdamW adds weight decay to Adam for better generalization.
            Recommended for most training jobs. Start with lr 1e-4 to 3e-4.
          </li>
          <li>
            <strong>AdamW Fused</strong> — Memory-efficient fused kernel variant of AdamW. Same behavior as AdamW but with
            lower memory usage.
          </li>
          <li>
            <strong>Adam8 / AdamW8</strong> — 8-bit versions of Adam/AdamW. Reduced memory footprint at minor precision
            cost. Good for larger models.
          </li>
          <li>
            <strong>AdamW FP8 / BF16</strong> — Custom 8-bit/BF16 AdamW implementations for specific hardware or precision
            requirements.
          </li>
          <li>
            <strong>Lion</strong> — Optimized for lower memory usage than Adam. Performs similarly to AdamW but with
            reduced memory overhead. Recommended for memory-constrained setups.
          </li>
          <li>
            <strong>Lion8Bit</strong> — 8-bit Lion variant. Same benefits as Lion with even lower memory usage.
          </li>
          <li>
            <strong>Adagrad</strong> — Adaptive learning rate per parameter. Useful for sparse gradients but generally
            less popular for diffusion training.
          </li>
          <li>
            <strong>Adafactor</strong> — Memory-efficient alternative to Adam that uses factored second-moment estimates.
            Good for large models when memory is a concern. Requires an LR scheduler.
          </li>
        </ul>
        <br />
        <strong>Adaptive Optimizers (built-in LR scheduling, no external scheduler needed):</strong>
        <ul className="list-disc pl-5 mt-2 space-y-1">
          <li>
            <strong>DAdaptAdam / DAdaptLion</strong> — Automatically determines the optimal learning rate based on
            parameter norms. No need to tune lr manually — just set lr to 1.0 and let it adapt. Excellent for
            fine-tuning when you don't want to search for the right learning rate.
          </li>
          <li>
            <strong>Prodigy / Prodigy8Bit</strong> — Combines DAdaptation with Adam/AdamW. Automatically scales the
            learning rate while using momentum. Great out-of-the-box performance without lr tuning.
          </li>
          <li>
            <strong>Automagic / Automagic v2 / Automagic v3</strong> — Self-adjusting optimizers that modify the learning
            rate based on update direction consistency. Automagic v3 uses a polarity window approach for more stable
            lr adaptation. No external scheduler needed.
          </li>
        </ul>
        <br />
        <strong>Recommendation:</strong> Start with <strong>AdamW8Bit</strong> for most cases. Use <strong>DAdaptAdam</strong>
        or <strong>Prodigy</strong> if you don't want to tune the learning rate manually. Use <strong>Lion</strong> if you
        are running low on VRAM.
      </>
    ),
  },

  // --- Scheduler docs ---
  lr_scheduler: {
    title: 'LR Scheduler',
    description: (
      <>
        A learning rate scheduler adjusts the learning rate during training according to a schedule. This can help the model
        converge better and avoid getting stuck in local minima.
        <br />
        <br />
        <strong>Note:</strong> Adaptive optimizers (DAdaptAdam, Prodigy, Automagic family) do not need an external LR
        scheduler — they handle learning rate adaptation internally. The scheduler will be locked to <em>None</em> when
        these are selected.
        <br />
        <br />
        <strong>Available Schedulers:</strong>
        <ul className="list-disc pl-5 mt-2 space-y-1">
          <li>
            <strong>None</strong> — No scheduling. The learning rate stays constant throughout training. Fine for simple
            cases or when using adaptive optimizers.
          </li>
          <li>
            <strong>Cosine</strong> — Smoothly decays the learning rate following a cosine curve. The most popular and
            generally recommended default. Works well for most training scenarios. Total Iters defaults to training steps
            divided by gradient accumulation steps.
          </li>
          <li>
            <strong>Cosine with Restarts</strong> — Like Cosine but resets the schedule periodically, allowing the model
            to escape local minima. Useful for long training runs. T_0 is the initial period, T_mult multiplies the
            period after each restart.
          </li>
          <li>
            <strong>Step</strong> — Multiplicatively reduces the learning rate by a factor every N steps. Simple and
            predictable. Step Size controls how often to reduce, Gamma controls the reduction factor (e.g., 0.1 means
            lr becomes 10% after each step).
          </li>
          <li>
            <strong>Polynomial</strong> — Decays the learning rate using a polynomial function. More aggressive decay
            than cosine at the end. Power controls the curve shape.
          </li>
          <li>
            <strong>Constant</strong> — Keeps the learning rate at a constant factor of the initial value. Useful when
            you want a fixed proportional reduction.
          </li>
          <li>
            <strong>Linear</strong> — Linearly decays the learning rate from start_factor to end_factor. Simple and
            predictable.
          </li>
          <li>
            <strong>Constant with Warmup</strong> — Starts with a linear warmup phase (ramping up the lr from 0 to the
            target) then holds it constant. Recommended for large models or when training from scratch. Warmup Steps
            defaults to 1000.
          </li>
        </ul>
        <br />
        <strong>Recommendation:</strong> <strong>Cosine</strong> is the best default for most training jobs. Use
        <strong>Constant with Warmup</strong> for large models or training from scratch. Use <strong>Cosine with
        Restarts</strong> for very long training runs.
      </>
    ),
  },
  // ============================================================================
  // Rank Gate Annealing Documentation
  // ============================================================================
  'config.process[0].network.rank_gates.enabled': {
    title: 'Enable Rank Gates',
    description: (
      <>
        Enable or disable SparseForge-inspired rank gate annealing. When enabled, each LoRA rank gets a soft gate that
        gradually prunes redundant ranks during training.
        <br />
        <br />
        <strong>Default:</strong> Enabled (recommended for all LoRA training).
        <br />
        <strong>Effect:</strong> Prevents rank collapse by making soft, curvature-aware pruning decisions instead of
        hard truncation.
      </>
    ),
  },
  'config.process[0].network.rank_gates.target_rank_ratio': {
    title: 'Target Rank Ratio',
    description: (
      <>
        Final fraction of ranks to keep after annealing. Gates anneal from 1 → {0,1}, selecting which ranks survive.
        <br />
        <br />
        <strong>Conservative (0.6–0.9):</strong> Keep more ranks. Use for small datasets, early exploration, or when
        you want to preserve rank diversity.
        <br />
        <strong>Aggressive (0.2–0.4):</strong> Keep fewer ranks. Use for large datasets, quick pruning of
        noise-dominated ranks, or when many ranks are redundant.
        <br />
        <strong>Default:</strong> 0.3 (keep 30% of ranks, prune 70%).
        <br />
        <br />
        <em>Example:</em> With rank 256 and ratio 0.3, ~77 ranks will survive at the end of training.
      </>
    ),
  },
  'config.process[0].network.rank_gates.lambda_mid_max': {
    title: 'Lambda Mid Max (Binary Preference)',
    description: (
      <>
        Maximum strength of the binary preference penalty <code>L_mid = Σ m(1-m)</code>. This penalty pushes gates
        toward 0 or 1 (decisive) rather than staying in the middle (0.5).
        <br />
        <br />
        <strong>Conservative (0.001–0.005):</strong> Weak penalty, gates can stay intermediate longer. Smoother
        transitions but slower pruning.
        <br />
        <strong>Aggressive (0.01–0.05):</strong> Strong penalty, forces gates to {0,1} quickly. Decisive pruning but
        less reversible.
        <br />
        <strong>Default:</strong> 0.01.
      </>
    ),
  },
  'config.process[0].network.rank_gates.alpha': {
    title: 'Alpha (Gate EMA Rate)',
    description: (
      <>
        EMA update rate for gates: <code>m ← (1-α)m + α·score</code>. Controls how quickly gates respond to
        curvature-aware importance scores.
        <br />
        <br />
        <strong>Conservative (0.01–0.05):</strong> Gentle updates, smooth transitions. Gates change slowly.
        <br />
        <strong>Aggressive (0.1–0.2):</strong> Fast updates, responsive to score changes. Gates adapt quickly.
        <br />
        <strong>Default:</strong> 0.1.
        <br />
        <br />
        <em>Tip:</em> Lower values (= smoother) work better for long training runs; higher values for short runs.
      </>
    ),
  },
  'config.process[0].network.rank_gates.gamma': {
    title: 'Gamma (Temperature Decay)',
    description: (
      <>
        Temperature decay per update: <code>T ← γ·T</code>. Controls how quickly the sigmoid sharpens (makes
        decisions more decisive) over time.
        <br />
        <br />
        <strong>Conservative (0.98–0.999):</strong> Slow decay, soft decisions maintained longer. Gentler annealing.
        <br />
        <strong>Aggressive (0.93–0.96):</strong> Fast decay, sharp decisions sooner. Quicker pruning.
        <br />
        <strong>Default:</strong> 0.95.
        <br />
        <br />
        <em>Effect:</em> Temperature starts at 1.0 and decays geometrically. Lower temperature = sharper sigmoid =
        more decisive 0/1 choices.
      </>
    ),
  },
  'config.process[0].network.rank_gates.update_every': {
    title: 'Update Every (Steps)',
    description: (
      <>
        Number of steps between gate updates. Gates are not updated every step to reduce overhead and avoid
        instability.
        <br />
        <br />
        <strong>Conservative (30–50):</strong> Infrequent updates. Smoother annealing but slower adaptation.
        <br />
        <strong>Aggressive (10–20):</strong> Frequent updates. Faster pruning but more computational overhead.
        <br />
        <strong>Default:</strong> 15.
        <br />
        <br />
        <em>Example:</em> With 2000 steps and update_every=15, gates update ~133 times. With update_every=25, only
        ~80 times.
      </>
    ),
  },
  'config.process[0].network.rank_gates.temperature': {
    title: 'Temperature (Initial)',
    description: (
      <>
        Initial sigmoid temperature for gate decisions. Higher temperature = softer/more uncertain decisions; lower =
        sharper/more confident.
        <br />
        <br />
        <strong>Conservative (2.0–5.0):</strong> Softer initial decisions, more exploration.
        <br />
        <strong>Aggressive (0.5–1.0):</strong> Sharper initial decisions, quicker pruning.
        <br />
        <strong>Default:</strong> 1.0.
        <br />
        <br />
        <em>Note:</em> Temperature decays over time via gamma, so initial value mainly affects early training.
      </>
    ),
  },
  'config.process[0].network.rank_gates.fisher_decay': {
    title: 'Fisher Decay (EMA)',
    description: (
      <>
        EMA decay for Fisher information diagonal estimation. Controls memory length of curvature tracking.
        <br />
        <br />
        <strong>Recommended:</strong> Keep at 0.999 for long memory. Lower values (0.95–0.99) react faster to
        gradient changes but are noisier.
        <br />
        <strong>Default:</strong> 0.999.
        <br />
        <br />
        <em>Why keep this default:</em> Fisher EMA needs long memory to accurately estimate curvature. Changing this
        is rarely beneficial.
      </>
    ),
  },
  'config.process[0].network.rank_gates.use_first_order': {
    title: 'Use First-Order Term',
    description: (
      <>
        Include the first-order term <code>|g·w|</code> in rank scoring. This adds gradient-weight magnitude to the
        curvature-based scoring.
        <br />
        <br />
        <strong>Recommended:</strong> Enabled (true). Helps distinguish between ranks with similar curvature but
        different gradient magnitudes.
        <br />
        <strong>Default:</strong> true.
      </>
    ),
  },
  'config.process[0].network.rank_gates.hardening_window': {
    title: 'Hardening Window',
    description: (
      <>
        Number of steps at the end of training for soft→hard interpolation. Gates gradually binarize during this
        window.
        <br />
        <br />
        <strong>Typical range:</strong> 200–1000 steps.
        <br />
        <strong>Default:</strong> 500 (auto-capped at 5% of total steps for short runs).
        <br />
        <br />
        <em>Effect:</em> Longer windows give smoother finalization; shorter windows make quicker final cuts.
      </>
    ),
  },
  'config.process[0].network.rank_gates.final_hardening': {
    title: 'Final Hardening (Binarize)',
    description: (
      <>
        Enable final binarization of gates at end of training. When enabled, gates are set to exactly 0 or 1 for the
        final saved model.
        <br />
        <br />
        <strong>Recommended:</strong> Enabled (true). Ensures the saved model has clean, discrete rank selection.
        <br />
        <strong>Default:</strong> true.
        <br />
        <br />
        <em>Note:</em> If disabled, gates remain continuous [0,1] in the saved model (useful for further fine-tuning).
      </>
    ),
  },
  'config.process[0].network.rank_gates.start_step': {
    title: 'Start Step (Annealing)',
    description: (
      <>
        Global step at which annealing begins. Leave empty for auto (5% of total steps).
        <br />
        <br />
        <strong>Auto:</strong> <code>max(100, total_steps × 0.05)</code>.
        <br />
        <strong>Manual:</strong> Set after warmup completes (typically 500–2000 steps).
        <br />
        <br />
        <em>Tip:</em> Don't start annealing during warmup; let weights stabilize first.
      </>
    ),
  },
  'config.process[0].network.rank_gates.end_step': {
    title: 'End Step (Annealing)',
    description: (
      <>
        Global step at which annealing completes. Leave empty for auto (75% of total steps).
        <br />
        <br />
        <strong>Auto:</strong> <code>min(total_steps - hardening_window, total_steps × 0.75)</code>.
        <br />
        <strong>Manual:</strong> Set before the hardening window begins.
        <br />
        <br />
        <em>Note:</em> Must be less than <code>total_steps - hardening_window</code>.
      </>
    ),
  },
  'config.process[0].network.rank_gates.eta_pen': {
    title: 'Eta Penalty (Mid-Preference Nudge)',
    description: (
      <>
        Penalty coefficient for nudging gates away from 0.5 (the "mid-preference" zone). Works in conjunction with
        <code>lambda_mid_max</code>.
        <br />
        <br />
        <strong>Typical range:</strong> 0.005–0.02.
        <br />
        <strong>Default:</strong> 0.01.
        <br />
        <br />
        <em>Effect:</em> Higher values push gates more aggressively away from indecisive 0.5 values.
      </>
    ),
  },
};

export const getDoc = (key: string | null | undefined): ConfigDoc | null => {
  if (key && key in docs) {
    return docs[key];
  }
  return null;
};

export default docs;
