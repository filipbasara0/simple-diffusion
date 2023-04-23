# Simple Denoising Diffusion

A minimal implementation of a denoising diffusion uncoditional image generation model in PyTorch.
The idea was to test the performance of a very small model on the Oxford Flowers dataset.

Includes the DDIM scheduler and the UNet architecture with residual connections and Attention layers.


## Oxford Flowers

![flowers](https://user-images.githubusercontent.com/29043871/197328106-97e825b5-814d-495c-9042-17e9962f9584.jpeg)


So far, the model was tested on the Oxford Flowers dataset - the results can be seen on the image above.
Images were generated with 50 DDIM steps.


The results were surprisingly decent and training unexpectedly smooth, considering the model size.

Training was done for `40k steps`, with a batch size of `64`. Learning rate was `1e-3` and weight decay was `5e-2`. Training took ~6 hours on GTX 1070Ti.

Hidden dims of `[16, 32, 64, 128]` were used, which resulted in a total of `2,346,835` million params.

To train the model, run the following command:
```
 python train.py   --dataset_name="huggan/flowers-102-categories"   --resolution=64   --output_dir="trained_models/ddpm-ema-pokemons-64.pth"   --train_batch_size=16   --num_epochs=121 --gradient_accumulation_steps=1   --learning_rate=1e-4   --lr_warmup_steps=300
```



### Conclusions
* Skip and residual connections are a must - training doesn't converge without them
* Attention speeds up convergence and improves the quality of generated samples
* Normalizing images to `N(0,1)` didn't yield improvents compared to the standard `-1 to 1` normalization
* Learning rate of `1e-3` resulted in a faster convergence for the smaller models, compared to `1e-4` which is usually used in literature


### Improvements
* Training longer - these models require a lot of iterations. For example, in [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/pdf/2105.05233.pdf), iterations ranged between `300K` and `4360k`!
* Using bigger models
* Would like to explore the impact of more diverse augmentations


### Future steps
* Training on `huggan/pokemons` dataset with a bigger model. This dataset proved to be too difficult for the `2M` model
* Training a model on a custom task
