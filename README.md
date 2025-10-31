<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h1 align="center">Towards Autoregressive Inverse Design of Disordered Metamaterials for Target Nonlinear Response</h1>

  <p align="center">
    Authors
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## Abstract

![Product Name Screen Shot][product-screenshot]

An insight is that metamaterials derive their novel mechanical properties from their unique architectures. Recent studies show adding disorder can expand the design space, but designing such structures for target nonlinear responses presents a major challenge, primarily in ensuring structural connectivity. This study introduces a data-driven framework to solve this problem, which pairs a generative autoregressive model for inverse design with a surrogate model for rapid forward prediction. The autoregressive model learns implicit connectivity rules from a large dataset and sequentially builds valid structures guided by a performance target. This method is demonstrated to successfully design disordered metamaterials with complex nonlinear properties on demand. An analysis of the model’s internal mechanism reveals its learned strategies for maintaining connectivity while approaching the target. This process involves a critical trade-off between design performance and structural integrity, which is controllable within this framework. The resulting method shows strong capabilities in both interpolation and extrapolation, generating novel designs that outperform examples from the training data. Overall, this work bridges the design of disordered metamaterials with the achievement of tailored nonlinear responses, opening a new path for creating high-performance functional materials. </br>
**Keywords:** Disordered Metamaterials, Autoregressive Model, Inverse Design, Nonlinear Response, Data-Driven Design, Deep Learning



<!-- GETTING STARTED -->
## Environment Setup

Clone this repository to your local machine, and install the dependencies.
  ```sh
  git clone git@github.com:dazhizhao/Autoregressive-Inverse-Design-for-Disordered-Metamaterials.git 
  pip install -r requirements.txt
  ```
## Dataset & Checkpoints
You can find the all the dataset and checkpoints we adopted in this paper from [Google Drive](https://drive.google.com/drive/folders/1lua95cX5zexfpzs-K3D9ZcxsPZSpGtCx?usp=drive_link). </br>
To remind you, our metamaterial structures are stored through `.mat` format and nonlinear responses are stored through `.xlxs` format. Plz prepare the correct software/tool to open it.
## Usage
### Autoregressive Transformer Model Inverse Design
#### Training Stage
To train the autoregressive transformer model, run this code:
```sh
  cd inverse
  python inverse.py
  ```
#### Inference Stage
Plz carefully check your own weight path `path_to_the_weight` below. </br>
For inversely designing the target obtained from disordered structures, run this code:
```sh
  cd inverse
  python post_process.py --ckpt_path path_to_the_weight --top_p 0.95 --temperature 1.0
  ```
For inversely designing the target obtained from periodic structures, run this code: 
```sh
  cd inverse
  python post_process_base.py --ckpt_path path_to_the_weight --top_p 0.95 --temperature 1.0
  ```
### Fourier Neural Operator Forward Prediction
#### Training Stage
To train the forward Fourier Neural Operator, run this code:
```sh
  cd forward
  python train_fno.py
  ```
#### Inference Stage
For forward predicting the nonlinear response from certain structures, run this code:
```sh
  cd forward
  python inference.py
  ```
## Citation
If you find our work helpful for you, plz cite and have a star for us! :blush:
```bibtex
@inproceedings{zhao2025ardisorder,
  title={Towards Autoregressive Inverse Design of Disordered Metamaterials for Target Nonlinear Response},
  author={Zhao, Dazhi and Xiang, Yujie and Zhang, Peng and Tang, Keke},
  booktitle={arXiv},
  year={2025}
}
```

## License
This project is licensed under the MIT License, see the LICENSE file for details.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/fig1_v2.png
[ar]: images/ar.png
[fno]: images/fno.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
