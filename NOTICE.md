# Licence and references

Code and weights (ours and Matthew Liew's): **CC-BY-NC 4.0**, full text in [LICENSE](LICENSE). The Apache 2.0 text for
the code derived from references 1-3 is in [external/LICENSE-APACHE](external/LICENSE-APACHE). Competition data:
[MIRA licence](http://rsna.org/mira-license); none of it is in this repository.

## Code

1. goodpjw2008. RSNA Knee 2.5D ConvNeXt MIL reader. Kaggle notebook
   [rsna-knee-coatnet-2-5d-convnext-mil-lb-0-950](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-2-5d-convnext-mil-lb-0-950)
   and dataset [rsna-knee-2-5d-convnext-reader](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader).
   Apache 2.0. Basis of Matthew's reader (`external/`).
2. heliosli. Kaggle notebook
   [0-9573-latest-public-dataset-score-in-one-click](https://www.kaggle.com/code/heliosli/0-9573-latest-public-dataset-score-in-one-click).
   Apache 2.0. Basis of the window members' inputs, model and training (`knee/windows.py`, the window family in
   `knee/models.py`, `knee/train.py`, `knee/infer.py`) and of the header knee centre in `external/reader/views.py`.
3. R. Wightman. PyTorch Image Models (timm), 2019. Apache 2.0.

Python packages are listed in [requirements.txt](requirements.txt). The inference notebook installs pylibjpeg,
pylibjpeg-openjpeg (MIT), pylibjpeg-libjpeg (GPLv3) and python-gdcm (BSD) from wheels in the code dataset, unmodified.

## Pretrained weights

4. Z. Liu et al. A ConvNet for the 2020s. CVPR 2022. timm `convnext_tiny.fb_in22k_ft_in1k_384` (crop-convnext,
   window-convnext); `convnext_tiny` (Matthew's encoder).
5. M. Tan, Q. Le. EfficientNetV2: Smaller Models and Faster Training. ICML 2021. timm `tf_efficientnetv2_s.in21k_ft_in1k`
   (lateral-effnet, window-effnet).
6. Label teachers only: timm `convnext_tiny.fb_in1k`, `coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k` (Z. Dai et al.,
   CoAtNet, NeurIPS 2021); RadImageNet ResNet-50 (X. Mei et al., Radiology: AI 2022). timm `resnet18`: Matthew's side
   classifier.

## Outside data (Matthew's members)

7. J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated MRI. arXiv:1811.08839, 2018; and
   F. Knoll et al. fastMRI: A Publicly Available Raw k-Space and DICOM Dataset of Knee Images. Radiology: AI 2020.
   Encoder pretraining and side classifier.
8. R. Zhao et al. fastMRI+: Clinical pathology annotations for knee and brain fully sampled MRI data. Scientific Data
   2022. Encoder pretraining targets.
9. Osteoarthritis Initiative (OAI), NIMH Data Archive; C. Peterfy et al., Osteoarthritis and Cartilage 2008. MOAKS
   grades: D. Hunter et al., Osteoarthritis and Cartilage 2011. pf-specialist's cartilage-grade head.

   The OAI is a public-private partnership comprised of five contracts (N01-AR-2-2258; N01-AR-2-2259;
   N01-AR-2-2260; N01-AR-2-2261; N01-AR-2-2262) funded by the National Institutes of Health, a branch of the
   Department of Health and Human Services, and conducted by the OAI Study Investigators. Private funding partners
   include Merck Research Laboratories; Novartis Pharmaceuticals Corporation, GlaxoSmithKline; and Pfizer, Inc.
   Private sector funding for the OAI is managed by the Foundation for the National Institutes of Health. This work
   was prepared using an OAI public use data set and does not necessarily reflect the opinions or views of the OAI
   investigators, the NIH, or the private funding partners.
