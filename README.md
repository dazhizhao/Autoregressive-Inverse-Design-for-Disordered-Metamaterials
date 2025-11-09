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

  <h1 align="center">Autoregressive Inverse Design of Disordered Metamaterials for Target Nonlinear Response</h1>

  <p align="center">
    Dazhi Zhao<sup>1</sup>, Yujie Xiang<sup>1</sup>, Peng Zhang<sup>1</sup>, Ning Liu<sup>1</sup>, Xianqiao Wang<sup>3</sup>, Keke Tang<sup>1,2*</sup> </br>
<sup>1</sup><em>School of Aerospace Engineering and Applied Mechanics, Tongji University, Shanghai 200092, China</em> </br>
<sup>2</sup><em>Key Laboratory of AI-aided Airworthiness of Civil Aircraft Structures, Civil Aviation Administration of China, Tongji University, Shanghai 200092, China</em> </br>
<sup>3</sup><em>School of ECAM, University of Georgia, Athens, GA 30602, USA</em> </br>
<sup>*</sup>Corresponding author: [kktang@tongji.edu.cn](kktang@tongji.edu.cn) (Keke Tang)

  </p>
</div>


<!-- ABOUT THE PROJECT -->
## Abstract

![Product Name Screen Shot][product-screenshot]

The pursuit of novel mechanical metamaterials faces a core dilemma: while introducing architectural disorder unlocks unprecedented design space, it jeopardizes the structural connectivity essential for mechanical integrity. We introduce a data-driven framework to resolve this critical trade-off in programming nonlinear responses. Our approach pairs a generative autoregressive model for inverse design with a surrogate model for rapid forward prediction. The autoregressive model learns implicit connectivity rules from data to sequentially build valid structures targeting specific performances. We demonstrate that this method successfully designs disordered metamaterials with target nonlinear properties on demand. Analysis of the model's internal mechanism reveals its learned strategies for maintaining connectivity, highlighting a critical yet controllable trade-off between performance and integrity. The framework exhibits strong interpolation and extrapolation capabilities, generating novel designs that outperform those in the training set. By providing a powerful and generalizable design tool, this work establishes a pathway to reliably meet complex functional requirements with disordered metamaterials, moving them from conceptual appeal toward practical viability.  </br>
**Keywords:** Mechanical Metamaterials, Architectural Disorder, Structural Connectivity, Autoregressive Model, Inverse Design, Nonlinear Response, Data-Driven Design



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
### Autoregressive Transformer Model for Inverse Design
If you wanna see more details about the autoregressive process, play the gif below:
![video][ar]
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
### Fourier Neural Operator for Forward Prediction
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
@article{zhao2025ardisorder,
  title={Autoregressive Inverse Design of Disordered Metamaterials for Target Nonlinear Response},
  author={Zhao, Dazhi and Xiang, Yujie and Zhang, Peng and Liu, Ning and Wang, Xianqiao and Tang, Keke},
  booktitle={arXiv},
  year={2025}
}
```
## Computing Resource Report
We report that all training and inference procedures for both models were conducted on a single NVIDIA RTX 4090 (24GB) GPU with the platform [Openbayes](https://openbayes.com/).

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
[ar]: images/video.gif
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
