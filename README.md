GIPMed - 234329

This project is taken in Geometric Image Processing - GIP lab of the CS faculty in Technion. 

Whole-slide imaging (WSIs), i.e. scanning and digitization of entire histology slides, are now being adopted across the world in pathology labs. Trained histopathologists can provide an accurate diagnosis of biopsy specimens based on WSI data. Given the dimensionality of WSIs and the increase in the number of potential cancer cases, analyzing these images is a time-consuming process. Automated segmentation of tissue helps in elevating the precision, speed, and reproducibility of research. Segmentation of WSI images is usually the precursor for performing various other downstream analyses such as classification and tumor burden estimation.

In this study, we investigate the use of deep learning models for the semantic segmentation of WSIs stained with Hematoxylin and Eosin images(H&E). H&E images are commonly used in pathology for the diagnosis and analysis of tissue samples, and automated segmentation can aid in the accurate and efficient identification of different structures within these images. This segmentation problem can be solved by classic computer vision algorithms, like Otsu’s thresholding method, however this method does not generalize well on different datasets and is not robust to artifacts on scans. Hence, we studied the use of deep learning models for this segmentation task, aiming to get better generalization ability and robustness to artifacts. We trained and tested several deep learning models, including U-Net and FusionNet, in a supervised manner on datasets of H&E images that include foreground tissue segmentation performed by the OTSU’s thresholding method. We evaluated their performance using various metrics, including visual examination. Our results show that these models can achieve high accuracy in the semantic segmentation of H&E images, and they generalize well on different datasets along with being robust to scan artifacts, making them promising tools for use in this segmentation task.

---------------------------------------------------------------------------------------------

For trainig run:
sbatch run.sh experiment.py --model-name <name_of_your_choice> [--model-type unet or fusionnet] [--datasets [list of datasets]]
                     [--add-artifacts] [--data-size DATA_SIZE] [--num-epochs NUM_EPOCHS]
                     [--early-stopping EARLY_STOPPING] [--batch-size BATCH_SIZE] [--input-size INPUT_SIZE]
                     [--num-workers NUM_WORKERS] [--pin-memory] [--load-model LOAD_MODEL]


For inference run: 
usage: inference.py [--model-name MODEL_NAME] [--num-classes NUM_CLASSES]
                    [--test-dataset [list of datasets]] [--data-size DATA_SIZE]
                    [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS] [--visualize]