# Paper Denoiser
We had lots of black and white printed papers with wrinkles and hand writing all over them. I performed a test to find and train a model to remove the noises.
## Models
these models are implemented
* Vanilla
* Denoising AutoEncoder (DAE)
* U-Net
* Custom U-Net + DAE
* Attention U-Net

Vanilla model tends to overfit towards identity function and did not do much denoising.
DAE model had good denoising but lost some details especially with letters like (ش پ ث) where there were close dots.
Custom U-Net + DAE model is just U-Net with a skip connection after Up-Conv. it performed good both in denoising and reconstruction.

U-Net and Attention U-Net models were not tested.
## Training
The models were trained on a RTX 3090 Ti.
Our target dataset was a big collection of papers with noise.
To train the model to remove the noises, a clean initial dataset of white printed papers were augmented with gaussian noise, blue lines, and gray lines.
Our model's objective was to remove these extra noises.

A common problem was white overfitting since most of our dataset consisted of white background.
The models were trained using multiple losses. First we used weighted Mean Absoulute Error (MAE) biased towards darker colors.
The models were quick to reconstruct dark areas but lacked the delicacy in close small dark dots like in letters (پ ش چ).
Then a weighted MAE biased towards both black and white colors were used to fix this problem.
Finally the training ended with normal MAE as the loss to balance the colors.
## Web UI
A simple web UI with `bootstrap` is in static files. After running `flask.ipynb` notebook, we can upload an image and download the denoised image in the browser. When paired with `ngrok` we can demo the models without physical access to the server with the GPU.
