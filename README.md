# Audio-Vision Multimodal Review (Paper list)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/yyysjz1997/Awesome-AudioVision-Multimodal)
[![Visits Badge](https://badges.pufler.dev/visits/yyysjz1997/Awesome-AudioVision-Multimodal)](https://badges.pufler.dev/visits/yyysjz1997/Awesome-AudioVision-Multimodal)

A curated list of **Audio-Vision Multimodal** with awesome resources (paper, application, data, review, survey, etc.), which aims to comprehensively and systematically summarize the recent advances to the best of our knowledge.

We will continue to update this list with newest resources. If you found any missed resources (paper/code) or errors, please feel free to open an issue or make a pull request.

Also, there is a [MindMap](https://github.com/yyysjz1997/Awesome-AudioVision-Multimodal/blob/main/MindMap.pdf)  for all papers, which is more intuitive.

- [Audio-Vision Multimodal Review (Paper list)](#audio-vision-multimodal-review-paper-list)
	- [Audio-vision Machine learning problem](#audio-vision-machine-learning-problem)
			- [Audio-vision representation learning](#audio-vision-representation-learning)
			- [Audio-vision saliency detection](#audio-vision-saliency-detection)
			- [Cross-modal transfer learning](#cross-modal-transfer-learning)
	- [Audio-vision enhancement](#audio-vision-enhancement)
			- [Speech recognition](#speech-recognition)
			- [Speech source enhancement/separation](#speech-source-enhancementseparation)
			- [Speaker verification (active speaker detection)](#speaker-verification-active-speaker-detection)
			- [Sound source separation](#sound-source-separation)
			- [Emotion recognition](#emotion-recognition)
			- [Action detection](#action-detection)
			- [Face super-resolution/reconstruction](#face-super-resolutionreconstruction)
	- [Cross-modal perception](#cross-modal-perception)
			- [Cross-modal generation](#cross-modal-generation)
				- [Video generation](#video-generation)
				- [Mono sound generation](#mono-sound-generation)
				- [Spatial sound generation](#spatial-sound-generation)
				- [Environment generation](#environment-generation)
			- [Cross-modal retrieval](#cross-modal-retrieval)
	- [Audio-vision synchronous applications](#audio-vision-synchronous-applications)
			- [Audio-vision localization](#audio-vision-localization)
				- [Sound localization in videos](#sound-localization-in-videos)
				- [Audio-vision navigation](#audio-vision-navigation)
				- [Audio-vision Event Localization](#audio-vision-event-localization)
				- [Sounding object localization](#sounding-object-localization)
			- [Audio-vision Parsing](#audio-vision-parsing)
			- [Audio-vision Dialog](#audio-vision-dialog)
			- [Audio-vision correspondence/correlation](#audio-vision-correspondencecorrelation)
			- [Face and Audio Matching](#face-and-audio-matching)
			- [Audio-vision question answering](#audio-vision-question-answering)
	- [Public dataset](#public-dataset)
	- [Review and survey](#review-and-survey)
	- [Diffusion model](#diffusion-model)
	- [Citation](#citation)
			


## Audio-vision Machine learning problem

#### Audio-vision representation learning

* [ICCV-2017] Look, Listen and Learn

* [CVPR-2018] Seeing voices and hearing faces: Cross-modal biometric matching

* [NeurIPS-2018] Cooperative Learning of Audio and Video Models from Self-Supervised Synchronization

* [CVPR-2019] Deep multimodal clustering for unsupervised  audiovisual  learning

* [ICASSP-2019] Perfect match: Improved cross-modal embeddings for audio-visual synchronisation

* [2020] Self-supervised learning of visual speech features with audiovisual speech enhancement

* [NeurIPS-2020] Learning Representations from Audio-Visual Spatial Alignment

* [NeurIPS-2020] Self-Supervised Learning by Cross-Modal Audio-Video Clustering

* [NeurIPS-2020] Labelling Unlabelled Videos From Scratch With Multi-Modal Self-Supervision

* [ACM MM-2020] Look, listen, and attend: Co-attention network for self-supervised audio-visual representation learning

* [CVPR-2021] Audio-Visual Instance Discrimination with Cross-Modal Agreement

* [CVPR-2021] Robust Audio-Visual Instance Discrimination

* [2021] Unsupervised Sound Localization via Iterative Contrastive Learning

* [ICCV-2021] Multimodal Clustering Networks for Self-Supervised Learning From Unlabeled Videos

* [IJCAI-2021] Speech2talking-face: Inferring and driving a face with synchronized audio-visual representation
  
* [2021] OPT: Omni-Perception Pre-Trainer for Cross-Modal Understanding and Generation

* [NeurIPS-2021] VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text

* [2021] Audio-visual Representation Learning for Anomaly Events Detection in Crowds

* [ICASSP-2022] Audioclip: Extending Clip to Image, Text and Audio

* [CVPR-2022] MERLOT Reserve: Neural Script Knowledge Through Vision and Language and Sound

* [2022] Probing Visual-Audio Representation for Video Highlight Detection via Hard-Pairs Guided Contrastive Learning

* [NeurIPS-2022] Non-Linguistic Supervision for Contrastive Learning of Sentence Embeddings

* [IEEE TMM-2022] Multimodal Information Bottleneck: Learning Minimal Sufficient Unimodal and Multimodal Representations

* [CVPR-2022] Audiovisual Generalised Zero-shot Learning with Cross-modal Attention and Language

* [CVPRW-2022] Multi-task Learning for Human Affect Prediction with Auditory–Visual Synchronized Representation

* [CVPR-2023] Vision Transformers are Parameter-Efficient Audio-Visual Learners

* [CVPR-2022] Audio-visual Generalised Zero-shot Learning with Cross-modal Attention and Language

* [ECCV-2022] Temporal and cross-modal attention for audio-visual zero-shot learning

* [Int. J. SpeechTechnol.-2022] Complementary models for audio-visual speech classification

* [Appl. Sci.-2022] Data augmentation for audio-visual emotion recognition with an efficient multimodal conditional GAN

* [NeurIPS-2022] u-HuBERT: Unified Mixed-Modal Speech Pretraining And Zero-Shot Transfer to Unlabeled Modality

* [NeurIPS-2022] Scaling Multimodal Pre-Training via Cross-Modality Gradient Harmonization

* [IEEE TMM-2022] Audiovisual tracking of concurrent speakers

* [AAAI-2023] Self-Supervised Audio-Visual Representation Learning with Relaxed Cross-Modal Synchronicity

* [ICLR-2023] Contrastive Audio-Visual Masked Autoencoder

* [ICLR-2023] Jointly Learning Visual and Auditory Speech Representations from Raw Data

* [WACV-2023] Audio Representation Learning by Distilling Video as Privileged Information

* [2023] AV-data2vec: Self-supervised Learning of Audio-Visual Speech Representations with Contextualized Target Representations

* [AAAI-2023] Audio-Visual Contrastive Learning with Temporal Self-Supervision

* [CVPR-2023] ImageBind One Embedding Space to Bind Them All

* [CVPR-2021] Spoken moments: Learning joint audio-visual representations from video descriptions

* [2020] Avlnet: Learning audiovisual language representations from instructional videos

* [ECCV-2018] Audio-visual scene analysis with selfsupervised multisensory features

* [MICCAI-2020] Self-Supervised Contrastive Video-Speech Representation Learning for Ultrasound

* [ECCV-2018] Jointly discovering visual objects and spoken words from raw sensory input

* [AAAI-2021] Enhancing Audio-Visual Association with Self-Supervised Curriculum Learning

* [ICLR-2019] Learning hierarchical discrete linguistic units from visually-grounded speech

* [2020] Learning speech representations from raw audio by joint audiovisual self-supervision

* [ICASSP-2020] Visually Guided Self Supervised Learning of Speech Representations

* [ECCV-2020] Leveraging  acoustic  images  for  effective  self-supervised  audio  representation  learning

#### Audio-vision saliency detection

* [2019] DAVE: A Deep Audio-Visual Embedding for Dynamic Saliency Prediction

* [CVPR-2020] STAViS: Spatio-Temporal AudioVisual Saliency Network

* [IEEE TIP-2020] A Multimodal Saliency Model for Videos With High Audio-visual Correspondence

* [IROS-2021] ViNet: Pushing the limits of Visual Modality for Audio-Visuav Saliency Prediction

* [CVPR-2021] From Semantic Categories to Fixations: A Novel Weakly-Supervised Visual-Auditory Saliency Detection Approach

* [ICME-2021] Lavs: A Lightweight Audio-Visual Saliency Prediction Model

* [2022] A Comprehensive Survey on Video Saliency Detection with Auditory Information: the Audio-visual Consistency Perceptual is the Key!

* [TOMCCAP-2022] PAV-SOD: A New Task Towards Panoramic Audiovisual Saliency Detection

* [CVPR-2023] CASP-Net: Rethinking Video Saliency Prediction from an Audio-VisualConsistency Perceptual Perspective

* [CVPR-2023] Self-Supervised Video Forensics by Audio-Visual Anomaly Detection

* [CVPR-2023] CASP-Net: Rethinking Video Saliency Prediction From an Audio-Visual Consistency Perceptual Perspective

* [IJCNN-2023] 3DSEAVNet: 3D-Squeeze-and-Excitation Networks for Audio-Visual Saliency Prediction

* [IEEE TMM-2023] SVGC-AVA: 360-Degree Video Saliency Prediction with Spherical Vector-Based Graph Convolution and Audio-Visual Attention

* [2022] Functional brain networks underlying auditory saliency during naturalistic listening experience

* [ICME-2021] Lavs: A lightweight audio-visual saliency prediction model

* [ICIP-2021] Deep audio-visual fusion neural network for saliency estimation

* [CVPR-2020] STAViS: Spatio–temporal audiovisual saliency network

* [2021] ViNet: Pushing the limits of visual modality for audiovisual saliency prediction

* [2021] Audiovisual saliency prediction via deep learning

* [CVPR-2019] Multi-source weak supervision for saliency detection

#### Cross-modal transfer learning

* [NeurIPS-2016] SoundNet: Learning Sound Representations from Unlabeled Video

* [ICCV-2019] Self-Supervised Moving Vehicle Tracking With Stereo Sound

* [CVPR-2021] There Is More Than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking With Sound by Distilling Multimodal Knowledge

* [AAAI-2021] Enhanced Audio Tagging via Multi* to Single-Modal Teacher-Student Mutual Learning

* [Interspeech-2021] Knowledge Distillation from Multi-Modality to Single-Modality for Person Verification

* [ICCV-2021] Multimodal Knowledge Expansion

* [CVPR-2021] Distilling Audio-visual Knowledge by Compositional Contrastive Learning

* [2022] Estimating Visual Information From Audio Through Manifold Learning

* [DCASE-2021] Audio-Visual Scene Classification Using A Transfer Learning Based Joint Optimization Strategy

* [Interspeech-2021] Audiovisual transfer learning for audio tagging and sound event detection

* [2023] Revisiting Pre-training in Audio-Visual Learning

* [IJCNN-2023] A Generative Approach to Audio-Visual Generalized Zero-Shot Learning: Combining Contrastive and Discriminative Techniques

* [ICCV-2023] Audio-Visual Class-Incremental Learning

* [ICCV-2023] Hyperbolic Audio-visual Zero-shot Learning

* [CVPR-2023] Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models

* [ICCV-2023] Class-Incremental Grouping Network for Continual Audio-Visual Learning

## Audio-vision enhancement 

#### Speech recognition

* [Applied Intelligence-2015] Audio-visual Speech Recognition Using Deep Learning

* [CVPR-2016] Temporal Multimodal Learning in Audiovisual Speech Recognition

* [AVSP-2017] End-To-End Audiovisual Fusion With LSTMs

* [IEEE TPAMI-2018] Deep Audio-visual Speech Recognition

* [ICASSP-2018] End-to-End Audiovisual Speech Recognition

* [2019] Explicit Sparse Transformer: Concentrated Attention Through Explicit Selection

* [ICASSP-2019] Modality attention for end-to-end audio-visual speech recognition

* [CVPR-2020] Discriminative Multi-Modality Speech Recognition

* [ICASSP0-2021] End-to-end audiovisual speech recognition with conformers

* [IEEE TNNLS-2022] Multimodal Sparse Transformer Network for Audio-visual Speech Recognition

* [Interspeech-2022] Robust Self-Supervised Audio-visual Speech Recognition

* [2022] Bayesian Neural Network Language Modeling for Speech Recognition

* [Interspeech-2022] Visual Context-driven Audio Feature Enhancement for Robust End-to-End Audio-Visual Speech Recognition

* [MLSP-2022] Rethinking Audio-visual Synchronization for Active Speaker Detection

* [NeurIPS-2022] A Single Self-Supervised Model for Many Speech Modalities Enables Zero-Shot Modality Transfer

* [ITOEC-2022] FSMS: An Enhanced Polynomial Sampling Fusion Method for Audio-Visual Speech Recognition

* [IJCNN-2022] Continuous Phoneme Recognition based on Audio-Visual Modality Fusion

* [ICIP-2022] Learning Contextually Fused Audio-Visual Representations For Audio-Visual Speech Recognition

* [ICASSP-2023] Self-Supervised Audio-Visual Speech Representations Learning By Multimodal Self-Distillation

* [CVPR-2022] Improving Multimodal Speech Recognition by Data Augmentation and Speech Representations

* [AAAI-2022] Distinguishing Homophenes Using Multi-Head Visual-Audio Memory for Lip Reading

* [AAAI-2023] Leveraging Modality-specific Representations for Audio-visual Speech Recognition via Reinforcement Learning

* [WACV-2023] Audio-Visual Efficient Conformer for Robust Speech Recognition

* [2023] Prompt Tuning of Deep Neural Networks for Speaker-adaptive Visual Speech Recognition

* [2023] Multimodal Speech Recognition for Language-Guided Embodied Agents

* [2023] MuAViC: A Multilingual Audio-Visual Corpus for Robust Speech Recognition and Robust Speech-to-Text Translation

* [ICASSP-2023] The NPU-ASLP System for Audio-Visual Speech Recognition in MISP 2022 Challenge

* [CVPR-2023] Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring

* [ICASSP-2023] Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels

* [CVPR-2023] AVFormer: Injecting Vision into Frozen Speech Models for Zero-Shot AV-ASR

* [CVPR-2023] SynthVSR: Scaling Up Visual Speech Recognition With Synthetic Supervision

* [ICASSP-2023] Multi-Temporal Lip-Audio Memory for Visual Speech Recognition

* [ICASSP-2023] On the Role of LIP Articulation in Visual Speech Perception

* [ICASSP-2023] Practice of the Conformer Enhanced Audio-Visual Hubert on Mandarin and English

* [ICASSP-2023] Robust Audio-Visual ASR with Unified Cross-Modal Attention

* [IJCAI-2023] Cross-Modal Global Interaction and Local Alignment for Audio-Visual Speech Recognition

* [Interspeech-2023] Prompting the Hidden Talent of Web-Scale Speech Models for Zero-Shot Task Generalization

* [Interspeech-2023] Improving the Gap in Visual Speech Recognition Between Normal and Silent Speech Based on Metric Learning

* [ACL-2023] AV-TranSpeech: Audio-Visual Robust Speech-to-Speech Translation

* [ACL-2023] Hearing Lips in Noise: Universal Viseme-Phoneme Mapping and Transfer for Robust Audio-Visual Speech Recognition

* [ACL-2023] MIR-GAN: Refining Frame-Level Modality-Invariant Representations with Adversarial Network for Audio-Visual Speech Recognition

* [IJCNN-2023] Exploiting Deep Learning for Sentence-Level Lipreading

* [IJCNN-2023] GLSI Texture Descriptor Based on Complex Networks for Music Genre Classification

* [ICME-2023] Improving Audio-Visual Speech Recognition by Lip-Subword Correlation Based Visual Pre-training and Cross-Modal Fusion Encoder

* [ICME-2023] Multi-Scale Hybrid Fusion Network for Mandarin Audio-Visual Speech Recognition

* [2023] A Review of Recent Advances on Deep Learning Methods for Audio-Visual Speech Recognition

* [EUSIPCO-2022] Visual Speech Recognition in a Driver Assistance System

* [Interspeech-2023] DAVIS: Driver’s Audio-Visual Speech Recognition

* [ICASSP-2020] Audio-Visual Recognition of Overlapped Speech for the LRS2 Dataset

* [ICCV-2023] SynthVSR: Scaling Up Visual Speech Recognition With Synthetic Supervision

* [CVPR-2021] Learning from the master: Distilling cross-modal advanced knowledge for lip reading
  
* [AAAI-2022] Crossmodal mutual learning for audio-visual speech recognition and manipulation

* [AAAI-2020] Hearing lips: Improving lip reading by distilling speech recognizers

* [ACMMM-2020] A lip sync expert is all you need for speech to lip generation in the wild

* [ACMMM-2021] Cross-modal selfsupervised learning for lip reading: When contrastive learning meets adversarial training

* [ICASSP-2020] Asr is all you need: Crossmodal distillation for lip reading

* [2021] Sub-word level lip reading with visual attention

* [2019] Improving audio-visual speech recognition performance with crossmodal student-teacher training
* [ICASSP-2021] End-to-end audio-visual speech recognition with conformers

#### Speech source enhancement/separation

* [IEEE/ACM Trans. Audio, Speech, Lang. Proces-2018] Using visual speech information in masking methods for audio speaker separation

* [Interspeech-2018] DNN driven speaker independent audio-visual mask estimation for speech separation

* [ICASSP-2018] Seeing through noise: Visually driven speaker separation and enhancement

* [ACM Trans on Graphics-2018] Looking  to listen at the cocktail party: A speaker-independent audiovisual model for speech separation

* [Interspeech-2018] Visual Speech Enhancement

* [Interspeech-2018] The Conversation: Deep Audio-Visual Speech Enhancement

* [IEEE TETCI-2018] Audio-Visual Speech Enhancement Using Multimodal Deep Convolutional Neural Networks

* [ICASSP-2018] Seeing Through Noise: Visually Driven Speaker Separation And Enhancement

* [ICASSP-2019] Face landmark-based speaker-independent  audio-visual  speech  enhancement  in  multi-talker  environments

* [IEEE/ACM Trans. Audio, Speech, Lang. Process-2019] Audio-visual deep clustering for speech separation

* [2019] Mixture of inference networks for VAE-based audio-visual speech enhancement

* [GlobalSIP-2019] Visually Assisted Time-Domain Speech Enhancement

* [ICASSP-2019] On Training Targets and Objective Functions for Deep-learning-based Audio-visual Speech Enhancement

* [InterSpeech-2019] Multimodal SpeakerBeam: Single Channel Target Speech Extraction with Audio-Visual Speaker Clues

* [ICASSP-2019] Effects of Lombard reflex on the performance of deep-learning-based audio-visual speech enhancement systems

* [Interspeech-2019] My Lips Are Concealed: Audio-Visual Speech Enhancement Through Obstructions

* [ICASSP-2020] A robust audio-visual speech enhancement model

* [2020] Facefilter: Audio-Visual Speech Separation Using Still Images

* [ICASSP-2020] A visual-pilot deep fusion for target speech separation in multitalker noisy environment

* [ICASSP-2020] Deep audio-visual speech separation with attention mechanism

* [ICASSP-2020] AV(SE)2: Audio-visual squeeze-excite speech enhancement

* [Interspeech-2020] Lite audio-visual speech enhancement

* [ICASSP-2020] Robust Unsupervised Audio-Visual Speech Enhancement Using a Mixture of Variational Autoencoders

* [CVPR-2021] Looking Into Your Speech: Learning Cross-Modal Affinity for Audio-Visual Speech Separation

* [ISCAS-2021] Audio-Visual Target Speaker Enhancement on Multi-Talker Environment using Event-Driven Cameras

* [ICASSP-2022] The Impact of Removing Head Movements on Audio-Visual Speech Enhancement

* [2022] Dual-path Attention is All You Need for Audio-Visual Speech Extraction

* [ICASSP-2022] Audio-visual multi-channel speech separation, dereverberation and recognition

* [2022] Audio-visual speech separation based on joint feature representation with cross-modal attention

* [CVPR-2022] Audio-Visual Speech Codecs: Rethinking Audio-Visual Speech Enhancement by Re-Synthesis

* [IEEE MMSP-2022] As We Speak: Real-Time Visually Guided Speaker Separation and Localization

* [IEEE HEALTHCOM-2022] A Novel Frame Structure for Cloud-Based Audio-Visual Speech Enhancement in Multimodal Hearing-aids

* [CVPR-2022] Reading to Listen at the Cocktail Party: Multi-Modal Speech Separation

* [WACV-2023] BirdSoundsDenoising: Deep Visual Audio Denoising for Bird Sounds

* [SLT-2023] AVSE Challenge: Audio-Visual Speech Enhancement Challenge

* [ICLR-2023] Filter-Recovery Network for Multi-Speaker Audio-Visual Speech Separation

* [WACV-2023] Unsupervised Audio-Visual Lecture Segmentation

* [ISCSLP-2022] Multi-Task Joint Learning for Embedding Aware Audio-Visual Speech Enhancement

* [ICASSP-2023] Real-Time Audio-Visual End-to-End Speech Enhancement

* [ICASSP-2023] Efficient Intelligibility Evaluation Using Keyword Spotting: A Study on Audio-Visual Speech Enhancement

* [ICASSP-2023] Incorporating Visual Information Reconstruction into Progressive Learning for Optimizing audio-visual Speech Enhancement

* [ICASSP-2023] Audio-Visual Speech Enhancement with a Deep Kalman Filter Generative Model

* [ICASSP-2023] A Multi-Scale Feature Aggregation Based Lightweight Network for Audio-Visual Speech Enhancement

* [ICASSP-2023] Egocentric Audio-Visual Noise Suppression

* [ICASSP-2023] Dual-Path Cross-Modal Attention for Better Audio-Visual Speech Extraction

* [ICASSP-2023] On the Role of Visual Context in Enriching Music Representations

* [ICASSP-2023] LA-VOCE: LOW-SNR Audio-Visual Speech Enhancement Using Neural Vocoders

* [ICASSP-2023] Learning Audio-Visual Dereverberation

* [Interspeech-2023] Incorporating Ultrasound Tongue Images for Audio-Visual Speech Enhancement through Knowledge Distillation

* [Interspeech-2023] Audio-Visual Speech Separation in Noisy Environments with a Lightweight Iterative Model

* [ITG-2023] Audio-Visual Speech Enhancement with Score-Based Generative Models

* [Interspeech-2023] Speech inpainting: Context-based speech synthesis guided by video

* [EUSIPCO-2023] Audio-Visual Speech Enhancement With Selective Off-Screen Speech Extraction

* [IEEE/ACM TASLP-2023] Audio-visual End-to-end Multi-channel Speech Separation, Dereverberation and Recognition

* [ICCV-2023] AdVerb: Visually Guided Audio Dereverberation

* [ICCV-2023] Position-Aware Audio-Visual Separation for Spatial Audio

* [ICCV-2023] Leveraging Foundation Models for Unsupervised Audio Visual Segmentation

* [2021] Visualvoice: Audio-visual speech separation with crossmodal consistency
* [2021] A cappella: Audio-visual singing voice separation

* [2020] Facefilter: Audiovisual speech separation using still images

* [2020] Deep variational generative models for audio-visual speech separation

#### Speaker verification (active speaker detection)

* [MTA-2016] Audio-visual Speaker Diarization Using Fisher Linear Semi-discriminant Analysis

* [ICASSP-2018] Audio-visual Person Recognition in Multimedia Data From the Iarpa Janus Program

* [ICASSP-2019] Noise-tolerant Audio-visual Online Person Verification Using an Attention-based Neural Network Fusion

* [Interspeech-2019] Who Said That?: Audio-visual Speaker Diarisation Of Real-World Meetings

* [ICASSP-2020] Self-Supervised Learning for Audio-visual Speaker Diarization

* [ICASSP-2020] The sound of my voice: Speaker representation loss for target voice separation

* [ICASSP-2021] A Multi-View Approach to Audio-visual Speaker Verification

* [IEEE/ACM TASLP-2021] Audio-visual Deep Neural Network for Robust Person Verification

* [ICDIP 2022] End-To-End Audiovisual Feature Fusion for Active Speaker Detection

* [EUVIP-2022] Active Speaker Recognition using Cross Attention Audio-Video Fusion

* [2022] Audio-Visual Activity Guided Cross-Modal Identity Association for Active Speaker Detection

* [SLT-2023] Push-Pull: Characterizing the Adversarial Robustness for Audio-Visual Active Speaker Detection

* [ICAI-2023] Speaker Recognition in Realistic Scenario Using Multimodal Data

* [CVPR-2023] A Light Weight Model for Active Speaker Detection

* [ICASSP-2023] The Multimodal Information based Speech Processing (MISP) 2022 Challenge: Audio-Visual Diarization and Recognition

* [ICASSP-2023] ImagineNet: Target Speaker Extraction with Intermittent Visual Cue Through Embedding Inpainting

* [ICASSP-2023] Speaker Recognition with Two-Step Multi-Modal Deep Cleansing

* [ICASSP-2023] Audio-Visual Speaker Diarization in the Framework of Multi-User Human-Robot Interaction

* [ICASSP-2023] Cross-Modal Audio-Visual Co-Learning for Text-Independent Speaker Verification

* [ICASSP-2023] Multi-Speaker End-to-End Multi-Modal Speaker Diarization System for the MISP 2022 Challenge

* [ICASSP-2023] Av-Sepformer: Cross-Attention Sepformer for Audio-Visual Target Speaker Extraction

* [ICASSP-2023] The WHU-Alibaba Audio-Visual Speaker Diarization System for the MISP 2022 Challenge

* [ICASSP-2023] Self-Supervised Audio-Visual Speaker Representation with Co-Meta Learning

* [Interspeech-2023] Target Active Speaker Detection with Audio-visual Cues

* [Interspeech-2023] CN-Celeb-AV: A Multi-Genre Audio-Visual Dataset for Person Recognition

* [Interspeech-2023] Rethinking the visual cues in audio-visual speaker extraction

* [ICAI-2023] Speaker Recognition in Realistic Scenario Using Multimodal Data

* [ACL-2023] OpenSR: Open-Modality Speech Recognition via Maintaining Multi-Modality Alignment

* [ICASSP-2023] AV-SepFormer: Cross-Attention SepFormer for Audio-Visual Target Speaker Extraction

* [Interspeech-2023] PIAVE: A Pose-Invariant Audio-Visual Speaker Extraction Network

* [NIPS-2018] Transfer learning from speaker verification to multispeaker text-to-speech synthesis

#### Sound source separation

* [IEEE Signal Process. Lett-2018] Listen and look: Audio-visual matching assisted speech source separation

* [ECCV-2018] Learning to Separate Object Sounds by Watching Unlabeled Video

* [IEEE Signal Processing Letters-2018] Listen  and  look: Audio–visual matching assisted speech source separation

* [ECCV-2018] The Sound of Pixels

* [ICASSP-2019] Self-supervised Audio-visual Co-segmentation

* [ICCV-2019] The Sound of Motions

* [ICCV-2019] Recursive Visual Sound Separation Using Minus-Plus Net

* [ICCV-2019] Co-Separating Sounds of Visual Objects

* [ACCV-2020] Visually Guided Sound Source Separation using Cascaded Opponent Filter Network

* [2020] Conditioned source separation for music instrument performances

* [CVPR-2020] Music Gesture for Visual Sound Separation

* [ICCV-2021] Visual Scene Graphs for Audio Source Separation

* [CVPR-2021] Cyclic Co-Learning of Sounding Object Visual Grounding and Sound Separation

* [ECCV-2022] AudioScopeV2: Audio-Visual Attention Architectures for Calibrated Open-Domain On-Screen Sound Separation

* [ICIP-2022] Visual Sound Source Separation with Partial Supervision Learning

* [NeurIPS-2022] Learning Audio-Visual Dynamics Using Scene Graphs for Audio Source Separation

* [ICLR-2023] CLIPSep: Learning Text-queried Sound Separation with Noisy Unlabeled Videos

* [CVPR-2023] Language-Guided Audio-Visual Source Separation via Trimodal Consistency

* [CVPR-2023] iQuery: Instruments As Queries for Audio-Visual Sound Separation

* [2020] Into the wild with audioscope: Unsupervised audio-visual separation of on-screen sounds

* [ECCV-2020] Self-supervised Learning of Audio-Visual Objects from Video

#### Emotion recognition

* [EMNLP-2017] Tensor Fusion Network for Multimodal Sentiment Analysis

* [AAAI-2018] Multi-attention Recurrent Network for Human Communication Comprehension

* [AAAI-2018] Memory Fusion Network for Multi-view Sequential Learning

* [NAACL-2018] Conversational Memory Network for Emotion Recognition in Dyadic Dialogue Videos

* [EMNLP-2018] Contextual Inter-modal Attention for Multi-modal Sentiment Analysis

* [IEEE Transactions on Affective Computing-2019] An  active learning  paradigm  for  online  audio-visual  emotion  recognition

* [ACL-2019] Multi-Modal Sarcasm Detection in Twitter with Hierarchical Fusion Model

* [ACL-2020] Sentiment and Emotion help Sarcasm? A Multi-task Learning Framework for Multi-Modal Sarcasm, Sentiment and Emotion Analysis

* [ACL-2020] A Transformer-based joint-encoding for Emotion Recognition and Sentiment Analysis

* [ACL-2020] Multilogue-Net: A Context Aware RNN for Multi-modal Emotion Detection and Sentiment Analysis in Conversation

* [CVPR-2021] Progressive Modality Reinforcement for Human Multimodal Emotion Recognition From Unaligned Multimodal Sequences

* [IEEE TAFFC-2021] Multi-modal Sarcasm Detection and Humor Classification in Code-mixed Conversations

* [IEEE SLT-2021] Detecting expressions with multimodal transformers

* [CVPR-2022] M2FNet: Multi-modal Fusion Network for Emotion Recognition in Conversation

* [CCC-2022] A Multimodal Emotion Perception Model based on Context-Aware Decision-Level Fusion

* [IJCNN-2022] Sense-aware BERT and Multi-task Fine-tuning for Multimodal Sentiment Analysis

* [Appl. Sci.-2022] Data augmentation for audio-visual emotion recognition with an efficient multimodal conditional GAN

* [IEEE/ACM TASLP-2022] EmoInt-Trans: A Multimodal Transformer for Identifying Emotions and Intents in Social Conversations

* [ICPR-2022] Self-attention fusion for audiovisual emotion recognition with incomplete data

* [IEEE TAFFC-2023] Audio-Visual Emotion Recognition With Preference Learning Based on Intended and Multi-Modal Perceived Labels

* [IEEE T-BIOM-2023] Audio-Visual Fusion for Emotion Recognition in the Valence-Arousal Space Using Joint Cross-Attention

* [ICASSP-2023] Adapted Multimodal Bert with Layer-Wise Fusion for Sentiment Analysis

* [ICASSP-2023] Recursive Joint Attention for Audio-Visual Fusion in Regression Based Emotion Recognition

* [IEEE/ACM TASLP-2023] Exploring Semantic Relations for Social Media Sentiment Analysis

* [CVPR-2023] Weakly Supervised Video Emotion Detection and Prediction via Cross-Modal Temporal Erasing Network

* [ACM MM-2023] Hierarchical Audio-Visual Information Fusion with Multi-label Joint Decoding for MER 2023

* [IJCNN-2019] Deep fusion: An attention guided factorized bilinear pooling for audio-video emotion recognition

* [2021] Does visual self-supervision improve learning of speech representations for emotion recognition

* [CVPR-2021] Audiodriven emotional video portraits

#### Action detection

* [IJCNN-2016] Exploring Multimodal Video Representation For Action Recognition

* [CVPR-2018] The ActivityNet Large-Scale Activity Recognition Challenge 2018 Summary

* [ICCV-2019] EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition

* [ICCV-2019] SCSampler: Sampling Salient Clips From Video for Efficient Action Recognition

* [ICCV-2019] Uncertainty-Aware Audiovisual Activity Recognition Using Deep Bayesian Variational Inference

* [CVPR-2020] Listen to Look: Action Recognition by Previewing Audio

* [2020] Audiovisual SlowFast Networks for Video Recognition

* [ICCV-2021] AdaMML: Adaptive Multi-Modal Learning for Efficient Video Recognition

* [2021] Cross-Domain First Person Audio-Visual Action Recognition through Relative Norm Alignment

* [WACV-2022] Domain Generalization Through Audio-Visual Relative Norm Alignment in First Person Action Recognition

* [CVPR-2022] Audio-Adaptive Activity Recognition Across Video Domains

* [WACV-2022] MM-ViT: Multi-Modal Video Transformer for Compressed Video Action Recognition

* [CVPR-2022] Learnable Irrelevant Modality Dropout for Multimodal Action Recognition on Modality-Specific Annotated Videos

* [2022] Noise-Tolerant Learning for Audio-Visual Action Recognition

* [ICLR-2023] Exploring Temporally Dynamic Data Augmentation for Video Recognition

* [ICASSP-2023] Epic-Sounds: A Large-scale Dataset of Actions That Sound

* [ICASSP-2023] AV-TAD: Audio-Visual Temporal Action Detection With Transformer

* [ICCV-2023] Audio-Visual Glance Network for Efficient Video Recognition

* [IEEE TMM-2023] Audio-Visual Contrastive and Consistency Learning for Semi-Supervised Action Recognition

* [Sensors-2023] Audio-Visual Speech and Gesture Recognition by Sensors of Mobile Devices

#### Face super-resolution/reconstruction

* [CVPR-2020] Learning to Have an Ear for Face Super-Resolution

* [IEEE TCSVT-2021] Appearance Matters, So Does Audio: Revealing the Hidden Face via Cross-Modality Transfer

* [CVPR-2022] Deep Video Inpainting Guided by Audio-Visual Self-Supervision

* [CVPR-2022] Cross-Modal Perceptionist: Can Face Geometry be Gleaned from Voices?

* [WACV-2023] Audio-Visual Face Reenactment

* [ICASSP-2023] Hearing and Seeing Abnormality: Self-Supervised Audio-Visual Mutual Learning for Deepfake Detection

* [CVPR-2023] AVFace: Towards Detailed Audio-Visual 4D Face Reconstruction

* [CVPR-2023] Parametric Implicit Face Representation for Audio-Driven Facial Reenactment

* [CVPR-2023] CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior

* [2021] One-shot talking face generation from single-speaker audio-visual correlation learning

* [CVPR-2021] Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset

* [NIPS-2019] Face  reconstruction  from voice  using  generative  adversarial  networks

## Cross-modal perception

#### Cross-modal generation

##### Video generation

* Generate face

	* [ACM TOG-2017] Synthesizing Obama: learning lip sync from audio

	* [ECCV-2018] Lip Movements Generation at a Glance

	* [IJCV-2019] You Said That?: Synthesising Talking Faces from Audio

    * [AAAI-2019] Talking face generation by adversarially disentangled audio-visual representation

	* [ICCV-2019] Few-Shot Adversarial Learning of Realistic Neural Talking Head Models

	* [IJCAI-2020] Arbitrary  talking  face  generation  via  attentional  audio-visual coherence  learning

	* [IJCV-2020] Realistic Speech-Driven Facial Animation with GANs

	* [IJCV-2020] GANimation: One-Shot Anatomically Consistent Facial Animation

	* [ACM TOG-2020] Makelttalk: Speaker-Aware Talking-Head Animation

	* [CVPR-2020] FReeNet: Multi-Identity Face Reenactment

	* [ECCV-2020] Neural Voice Puppetry: Audio-driven Facial Reenactment

	* [CVPR-2020] Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images

	* [ECCV-2020] MEAD: A Large-scale Audio-visual Dataset for Emotional Talking-face Generation

	* [AAAI-2021] Write-a-speaker: Text-based Emotional and Rhythmic Talking-head Generation

	* [CVPR-2021] Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation

	* [CVPR-2021] Audio-Driven Emotional Video Portraits

	* [AAAI-2022] One-shot Talking Face Generation from Single-speaker Audio-Visual Correlation Learning

	* [TVCG-2022] Generating talking face with controllable eye movements by disentangled blinking feature

	* [AAAI-2022] SyncTalkFace: Talking Face Generation with Precise Lip-Syncing via Audio-Lip Memory

	* [CVPR-2022] FaceFormer: Speech-Driven 3D Facial Animation with Transformers

	* [CVPR-2023] Seeing What You Said: Talking Face Generation Guided by a Lip Reading Expert

	* [ICASSP-2023] Free-View Expressive Talking Head Video Editing

	* [ICASSP-2023] Audio-Driven Facial Landmark Generation in Violin Performance using 3DCNN Network with Self Attention Model

	* [ICASSP-2023] Naturalistic Head Motion Generation from Speech

	* [ICASSP-2023] Audio-Visual Inpainting: Reconstructing Missing Visual Information with Sound

	* [CVPR-2023] Identity-Preserving Talking Face Generation with Landmark and Appearance Priors

	* [CVPR-2023] SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation

	* [ACM MM-2023] Hierarchical Semantic Perceptual Listener Head Video Generation: A High-performance Pipeline

	* [CVPR-2023] LipFormer: High-fidelity and Generalizable Talking Face Generation with A Pre-learned Facial Codebook
  
	* [IEEE TMM-2022] Audio-driven talking face video generation with dynamic convolution kernels

* Generate gesture

	* [IVA-2018] Evaluation of Speech-to-Gesture Generation Using Bi-Directional LSTM Network

	* [IVA-2019] Analyzing Input and Output Representations for Speech-Driven Gesture Generation

	* [CVPR-2019] Learning Individual Styles of Conversational Gesture

	* [ICMI-2019] To React or not to React: End-to-End Visual Pose Forecasting for Personalized Avatar during Dyadic Conversations,

	* [EUROGRAPHICS-2020] Style-Controllable Speech-Driven Gesture Synthesis Using Normalising Flows

	* [ICMI-2020] Gesticulator: A Framework For Semantically-Aware Speech-Driven Gesture Generation

	* [2020] Style Transfer for Co-Speech Gesture Animation: A Multi-Speaker Conditional-Mixture Approach

	* [ACM TOG-2020] Speech Gesture Generation From The Trimodal Context Of Text, Audio, And Speaker Identity

	* [CVPR-2022] SEEG: Semantic Energized Co-Speech Gesture Generation

	* [IEEE TNNLS-2022] VAG: A Uniform Model for Cross-Modal Visual-Audio Mutual Generation

	* [CVPR-2023] Taming Diffusion Models for Audio-Driven Co-Speech Gesture Generation

* Generate dance

	* [ACM MM-2018] Dance with Melody: An LSTM-autoencoder Approach to Music-oriented Dance Synthesis

	* [CVPR-2018] Audio to Body Dynamics

	* [NeurIPS-2019] Dancing to Music

	* [ICLR-2021] Dance Revolution: Long-Term Dance Generation with Music via Curriculum Learning

	* [ICCV-2021] AI Choreographer: Music Conditioned 3D Dance Generation With AIST++

	* [ICASSP-2022] Genre-Conditioned Long-Term 3D Dance Generation Driven by Music

	* [CVPR-2022] Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory

	* [CVPR-2023] MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation

	* [IEEE TMM-2023] Learning Music-Dance Representations through Explicit-Implicit Rhythm Synchronization

	* [ICCV-2023] Listen and Move: Improving GANs Coherency in Agnostic Sound-to-Video Generation

* Other

	* [CVPR-2023] Sound to Visual Scene Generation by Audio-to-Visual Latent Alignment

	* [2021] Sound-guided semantic image manipulation

	* [2022] Learning visual styles from audio-visual associations 

	* [ICCV-2023] Sound to Visual Scene Generation by Audio-to-Visual Latent Alignment
  
	* [2021] Speech2video: Cross-modal distillation for speech to video generation

    * [2018] CMCGAN: A uniform framework  for  cross-modal  visual-audio  mutual  generation

    * [IEEE/ACM Transactions on Audio, Speech, and Language Processing-2021] Generating  images  from  spoken  descriptions

##### Mono sound generation

* Speech

	* [ICASSP-2017] Vid2speech: Speech Reconstruction From Silent Video

	* [ICCV-2017] Improved Speech Reconstruction From Silent Video

	* [ICASSP-2018] Lip2Audspec: Speech Reconstruction from Silent Lip Movements Video

	* [ACM MM-2018] Harnessing AI for Speech Reconstruction using Multi-view Silent Video Feed

	* [Interspeech-2019] Video-Driven Speech Reconstruction using Generative Adversarial Networks

	* [Interspeech-2019] Hush-Hush Speak: Speech Reconstruction Using Silent Videos

	* [ICASSP-2021] Learning Audio-Visual Correlations From Variational Cross-Modal Generation

	* [IEEE TCYB-2022] End-to-End Video-to-Speech Synthesis Using Generative Adversarial Networks

	* [ICPR-2022] Learning Speaker-specific Lip-to-Speech Generation

	* [ICASSP-2023] Imaginary Voice: Face-styled Diffusion Model for Text-to-Speech

	* [CVPR-2023] ReVISE: Self-Supervised Speech Resynthesis With Visual Input for Universal and Generalized Speech Regeneration

	* [ICCV-2023] DiffV2S: Diffusion-based Video-to-Speech Synthesis with Vision-guided Speaker Embedding

	* [ICCV-2023] Let There Be Sound: Reconstructing High Quality Speech from Silent Videos

    * [CVPR-2019] Speech2Face:  Learning  the face behind a voice

* Music

	* [IEEE TMM-2015] Real-Time Piano Music Transcription Based on Computer Vision

	* [ACM MM-2017] Deep Cross-Modal Audio-Visual Generation

	* [NeurIPS-2020] Audeo: Audio Generation for a Silent Performance Video

	* [ECCV-2020] Foley Music: Learning to Generate Music from Videos

	* [ICASSP-2020] Sight to Sound: An End-to-End Approach for Visual Piano Transcription

	* [2020] Multi-Instrumentalist Net: Unsupervised Generation of Music from Body Movements

	* [ICASSP-2021] Collaborative Learning to Generate Audio-Video Jointly

	* [ACM-2021] Video Background Music Generation with Controllable Music Transformer

	* [2022] Vis2Mus: Exploring Multimodal Representation Mapping for Controllable Music Generation

	* [CVPR-2023] Conditional Generation of Audio from Video via Foley Analogies

	* [ICML-2023] Long-Term Rhythmic Video Soundtracker

*  Natural Sound

	* [CVPR-2016] Visually Indicated Sounds

	* [CVPR-2018] Visual to Sound: Generating Natural Sound for Videos in the Wild

	* [IEEE TIP-2020] Generating Visually Aligned Sound From Videos

	* [BMVC-2021] Taming Visually Guided Sound Generation

	* [IEEE TCSVT-2022] Towards an End-to-End Visual-to-Raw-Audio Generation With GAN

	* [ICASSP-2023] I Hear Your True Colors: Image Guided Audio Generation

	* [CVPR-2023] Physics-Driven Diffusion Models for Impact Sound Synthesis from Videos

##### Spatial sound generation

* [ACM TOG-2018] Scene-aware audio for 360° videos

* [NeurIPS-2018] Self-Supervised Generation of Spatial Audio for 360° Video

* [CVPR-2019] 2.5D Visual Sound

* [ICIP-2019] Self-Supervised Audio Spatialization with Correspondence Classifier

* [ECCV-2020] Sep-Stereo: Visually Guided Stereophonic Audio Generation by Associating Source Separation

* [CVPR-2021] Visually Informed Binaural Audio Generation without Binaural Audios

* [AAAI-2021] Exploiting Audio-Visual Consistency with Partial Supervision for Spatial Audio Generation

* [TOG-2021] Binaural Audio Generation via Multi-task Learning

* [WACV-2022] Beyond Mono to Binaural: Generating Binaural Audio From Mono Audio With Depth and Cross Modal Attention

* [CVPR-2023] Novel-View Acoustic Synthesis

* [ICCV-2023] Separating Invisible Sounds Toward Universal Audio-Visual Scene-Aware Sound Separation

##### Environment generation

* [ICRA-2020] BatVision: Learning to See 3D Spatial Layout with Two Ears

* [ECCV-2020] VISUALECHOES: Spatial Image Representation Learning Through Echolocation

* [CVPR-2021] Beyond Image to Depth: Improving Depth Prediction Using Echoes

* [ICASSP-2022] Co-Attention-Guided Bilinear Model for Echo-Based Depth Estimation

* [NeurIPS-2022] Learning Neural Acoustic Fields

* [NeurIPS-2022] Few-Shot Audio-Visual Learning of Environment Acoustics

#### Cross-modal retrieval

* [2017] Content-Based Video-Music Retrieval Using Soft Intra-Modal Structure Constraint

* [ICCV-2017] Image2song: Song Retrieval via Bridging Image Content and Lyric Words

* [CVPR-2018] Seeing voices and hearing faces: Cross-modal biometric matching

* [ECCV-2018] Cross-modal Embeddings for Video and Audio Retrieval

* [ISM-2018] Audio-Visual Embedding for Cross-Modal Music Video Retrieval through Supervised Deep CCA

* [TOMCCAP-2020] Deep Triplet Neural Networks with Cluster-CCA for Audio-Visual Cross-Modal Retrieval

* [IEEE TGRS-2020] Deep Cross-Modal Image–Voice Retrieval in Remote Sensing

* [2021] Learning Explicit and Implicit Latent Common Spaces for Audio-Visual Cross-Modal Retrieval

* [ICCV-2021] Temporal Cue Guided Video Highlight Detection With Low-Rank Audio-Visual Fusion

* [IJCAI-2022] Unsupervised Voice-Face Representation Learning by Cross-Modal Prototype Contrast

* [IEEE ISM-2022] Complete Cross-triplet Loss in Label Space for Audio-visual Cross-modal Retrieval

* [IEEE SMC-2022] Graph Network based Approaches for Multi-modal Movie Recommendation System

* [CVPR-2022] Visual Acoustic Matching

* [2021] Learning explicit and implicit latent common spaces for audio-visual cross-modal retrieval

* [2021] Variational autoencoder with cca for audio-visual cross-modal retrieval.

* [TMM-2021] Adversarial-metric learning for audio-visual cross-modal matching

## Audio-vision synchronous applications

#### Audio-vision localization

##### Sound localization in videos

* [ECCV-2018] Objects that Sound
  
* [CVPR-2018] Learning to localize sound source in visual scenes

* [ECCV-2018] Audio-Visual Scene Analysis with Self-Supervised Multisensory Features

* [ECCV-2018] The Sound of Pixels

* [ICASSP-2019] Self-supervised Audio-visual Co-segmentation

* [ICCV-2019] The Sound of Motions

* [CVPR-2019] Deep Multimodal Clustering for Unsupervised Audiovisual Learning

* [CVPR-2021] Localizing Visual Sounds the Hard Way

* [IEEE TPAMI-2021] Class-aware Sounding Objects Localization via Audiovisual Correspondence

* [IEEE TPAMI-2021] Learning to Localize Sound Sources in Visual Scenes: Analysis and Applications

* [CVPR-2022] Mix and Localize: Localizing Sound Sources in Mixtures

* [ECCV-2022] Audio-Visual Segmentation

* [2022] Egocentric Deep Multi-Channel Audio-Visual Active Speaker Localization

* [ACM MM-2022] Exploiting Transformation Invariance and Equivariance for Self-supervised Sound Localisation

* [CVPR-2022] Self-Supervised Predictive Learning: A Negative-Free Method for Sound Source Localization in Visual Scenes

* [CVPR-2022] Self-supervised object detection from audio-visual correspondence

* [EUSIPCO-2022] Visually Assisted Self-supervised Audio Speaker Localization and Tracking

* [ICASSP-2023] MarginNCE: Robust Sound Localization with a Negative Margin

* [IEEE TMM-2022] Cross modal video representations for weakly supervised active speaker localization

* [NeurIPS-2022] A Closer Look at Weakly-Supervised Audio-Visual Source Localization

* [AAAI-2022] Visual Sound Localization in the Wild by Cross-Modal Interference Erasing

* [ECCV-2022] Sound Localization by Self-Supervised Time Delay Estimation

* [IEEE/ACM TASLP-2023] Audio-Visual Cross-Attention Network for Robotic Speaker Tracking

* [WACV-2023] Hear The Flow: Optical Flow-Based Self-Supervised Visual Sound Source Localization

* [WACV-2023] Exploiting Visual Context Semantics for Sound Source Localization

* [2023] Audio-Visual Segmentation with Semantics

* [CVPR-2023] Learning Audio-Visual Source Localization via False Negative Aware Contrastive Learning

* [CVPR-2023] Egocentric Audio-Visual Object Localization

* [CVPR-2023] Learning Audio-Visual Source Localization via False Negative Aware Contrastive Learning

* [CVPR-2023] Audio-Visual Grouping Network for Sound Localization from Mixtures

* [ICASSP-2023] Flowgrad: Using Motion for Visual Sound Source Localization

* [ACM MM-2023] Audio-visual segmentation, sound localization, semantic-aware sounding objects localization

* [ACM MM-2023] Induction Network: Audio-Visual Modality Gap-Bridging for Self-Supervised Sound Source Localization

* [ACM MM-2023] Audio-Visual Spatial Integration and Recursive Attention for Robust Sound Source Localization

* [IJCAI-2023] Discovering Sounding Objects by Audio Queries for Audio Visual Segmentation

* [IROS-2020] Self-supervised Neural Audio-Visual Sound Source Localization via Probabilistic Spatial Modeling

##### Audio-vision navigation

* [ECCV-2020] SoundSpaces: Audio-Visual Navigation in 3D Environments

* [ICRA-2020] Look, Listen, and Act: Towards Audio-Visual Embodied Navigation

* [ICLR-2021] Learning to Set Waypoints for Audio-Visual Navigation

* [CVPR-2021] Semantic Audio-Visual Navigation

* [ICCV-2021] Move2Hear: Active Audio-Visual Source Separation

* [2022] Sound Adversarial Audio-Visual Navigation

* [CVPR-2022] Towards Generalisable Audio Representations for Audio-Visual Navigation

* [NeurIPS-2022] SoundSpaces 2.0: A Simulation Platform for Visual-Acoustic Learning

* [NeurIPS-2022] AVLEN: Audio-Visual-Language Embodied Navigation in 3D Environments

* [BMVC-2022] Pay Self-Attention to Audio-Visual Navigation

* [CVPR-2022] Finding Fallen Objects Via Asynchronous Audio-Visual Integration

* [CVPR-2022] ObjectFolder 2.0: A Multisensory Object Dataset for Sim2Real Transfer

* [IEEE RAL-2023] Catch Me If You Hear Me: Audio-Visual Navigation in Complex Unmapped Environments with Moving Sounds

* [2023] Audio Visual Language Maps for Robot Navigation

* [ICCV-2023] Omnidirectional Information Gathering for Knowledge Transfer-based Audio-Visual Navigation

##### Audio-vision Event Localization

* [CVPR-2018] Weakly  supervised  representation  learning for unsynchronized audio-visual events

* [ECCV-2018] Audio-visual Event Localization in Unconstrained Videos

* [ICASSP-2019] Dual-modality Seq2Seq Network for Audio-visual Event Localization

* [ICCV-2019] Dual Attention Matching for Audio-Visual Event Localization

* [2020] Crossmodal learning  for  audio-visual  speech  event  localization

* [AAAI-2020] Cross-Modal Attention Network for Temporal Inconsistent Audio-Visual Event Localization

* [ACCV-2020] Audiovisual Transformer with Instance Attention for Audio-Visual Event Localization

* [WACV-2021] Audio-Visual Event Localization via Recursive Fusion by Joint Co-Attention

* [CVPR-2021] Positive Sample Propagation along the Audio-Visual Event Line

* [AIKE-2021] Audio-Visual Event Localization based on Cross-Modal Interacting Guidance

* [TMM-2021] Audio-Visual Event Localization by Learning Spatial and Semantic Co-attention

* [CVPR-2022] Cross-Modal Background Suppression for Audio-Visual Event Localization

* [ICASSP-2022] Bi-Directional Modality Fusion Network For Audio-Visual Event Localization

* [ICSIP-2022] Audio-Visual Event and Sound Source Localization Based on Spatial-Channel Feature Fusion

* [IJCNN-2022] Look longer to see better: Audio-visual event localization by exploiting long-term correlation

* [EUSIPCO-2022] Audio Visual Graph Attention Networks for Event Detection in Sports Video

* [IEEE TPAMI-2022] Contrastive Positive Sample Propagation along the Audio-Visual Event Line

* [IEEE TPAMI-2022] Semantic and Relation Modulation for Audio-Visual Event Localization

* [WACV-2023] AVE-CLIP: AudioCLIP-based Multi-window Temporal Transformer for Audio Visual Event Localization

* [WACV-2023] Event-Specific Audio-Visual Fusion Layers: A Simple and New Perspective on Video Understanding

* [ICASSP-2023] A dataset for Audio-Visual Sound Event Detection in Movies

* [CVPR-2023] Dense-Localizing Audio-Visual Events in Untrimmed Videos: A Large-Scale Benchmark and Baseline

* [CVPR-2023] Collaborative Noisy Label Cleaner: Learning Scene-aware Trailers for Multi-modal Highlight Detection in Movies

* [ICASSP-2023] Collaborative Audio-Visual Event Localization Based on Sequential Decision and Cross-Modal Consistency

* [CVPR-2023] Collecting Cross-Modal Presence-Absence Evidence for Weakly-Supervised Audio-Visual Event Perception

* [IJCNN-2023] Specialty may be better: A decoupling multi-modal fusion network for Audio-visual event localization

* [AAAI-2023] Furnishing Sound Event Detection with Language Model Abilities

* [ICCV-2023] Prompting Segmentation with Sound is Generalizable Audio-Visual Source Localizer



##### Sounding object localization

* [CVPR-2018] Learning to localize sound source in visual scenes

* [IROS-2021] AcousticFusion: Fusing sound source localization to visual SLAM in dynamic environments

* [ITPAMI-2022] Classaware sounding objects localization via audiovisual correspondence

* [PR-2021]Multimodal fusion for indoor sound source localization

* [CVPR-2021] Localizing Visual Sounds the Hard Way

#### Audio-vision Parsing

* [ECCV-2020] Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video Parsing

* [CVPR-2021] Exploring Heterogeneous Clues for Weakly-Supervised Audio-Visual Video Parsing

* [NeurIPS-2021] Exploring Cross-Video and Cross-Modality Signals for Weakly-Supervised Audio-Visual Video Parsing

* [2022] Investigating Modality Bias in Audio Visual Video Parsing

* [ICASSP-2022] Distributed Audio-Visual Parsing Based On Multimodal Transformer and Deep Joint Source Channel Coding

* [ECCV-2022] Joint-Modal Label Denoising for Weakly-Supervised Audio-Visual Video Parsing

* [NeurIPS-2022] Multi-modal Grouping Network for Weakly-Supervised Audio-Visual Video Parsing

* [2023] Improving Audio-Visual Video Parsing with Pseudo Visual Labels

* [ICASSP-2023] CM-CS: Cross-Modal Common-Specific Feature Learning For Audio-Visual Video Parsing

* [2023] Towards Long Form Audio-visual Video Understanding

* [CVPR-2023] Collecting Cross-Modal Presence-Absence Evidence for Weakly-Supervised Audio* Visual Event Perception

#### Audio-vision Dialog

* [CVPR-2019] Audio Visual Scene-Aware Dialog

* [Interspeech-2019] Joint Student-Teacher Learning for Audio-Visual Scene-Aware Dialog

* [ICASSP-2019] End-to-end Audio Visual Scene-aware Dialog Using Multimodal Attention-based Video Features

* [CVPR-2019] A Simple Baseline for Audio-Visual Scene-Aware Dialog

* [CVPR-2019] Exploring Context, Attention and Audio Features for Audio Visual Scene-Aware Dialog

* [2020] TMT: A Transformer-based Modal Translator for Improving Multimodal Sequence Representations in Audio Visual Scene-aware Dialog

* [AAAI-2021] Dynamic Graph Representation Learning for Video Dialog via Multi-Modal Shuffled Transformers

* [2021] VX2TEXT: End-to-End Learning of Video-Based Text Generation From Multimodal Inputs

* [ICASSP-2022] Audio-Visual Scene-Aware Dialog and Reasoning Using Audio-Visual Transformers with Joint Student-Teacher Learning

* [WACV-2022] QUALIFIER: Question-Guided Self-Attentive Multimodal Fusion Network for Audio Visual Scene-Aware Dialog

* [TACL-2022] Learning English with Peppa Pig

* [2022] End-to-End Multimodal Representation Learning for Video Dialog

* [AAAI-2022] Audio Visual Scene-Aware Dialog Generation with Transformer-based Video Representations

* [IEEE/ACM TASLP-2023] DialogMCF: Multimodal Context Flow for Audio Visual Scene-Aware Dialog

#### Audio-vision correspondence/correlation

* [ICASSP-2019] Learning affective correspondence between music and image

* [2020] Themes informed audio-visual correspondence learning

* [CVPR-2021] Audio-Visual Instance Discrimination with Cross-Modal Agreement

* [Computer Vision and Image Understanding* 2023] Unsupervised sound localization via iterative contrastive learning

* [2021] Vatt: Transformers for multimodal selfsupervised learning from raw video, audio and text.

* [NeurIPS-2020] Self-supervised multimodal versatile networks

* [Neurocomputing-2020] Audio–visual domain adaptation using conditional semi-supervised generative adversarial networks

* [2021] Cross-modal attention consistency for video-audio unsupervised learning

* [ICCV-2023] Bi-directional Image-Speech Retrieval Through Geometric Consistency

#### Face and Audio Matching

* [2017] Putting a face to the voice: Fusing audio and visual signals across a video to determine speakers

* [CVPR-2018] Seeing voices and hearing faces: Cross-modal biometric matching

* [ECCV-2018] Learnable pins: Crossmodal embeddings for person identity

* [ICLR-2019] Disjoint mapping network for cross-modal matching of voices and faces

* [ICME-2019] A novel distance learning for elastic cross-modal audio-visual matching

* [ECCV-2018] Crossmodal embeddings for video and audio retrieval

#### Audio-vision question answering

* [ICCV-2021] Pano-AVQA: Grounded Audio-Visual Question Answering on 360deg Videos

* [CVPR-2022] Learning To Answer Questions in Dynamic Audio-Visual Scenarios

* [NeurIPS-2022] Language Models with Image Descriptors are Strong Few-Shot Video-Language Learners

* [ACM MM-2023] Progressive Spatio-temporal Perception for Audio-Visual Question Answering

## Public dataset

* LRW, LRS2 and LRS3 (Speech-related, speaker-related,face generation-related tasks)

* VoxCeleb, VoxCeleb2 (Speech-related, speaker-related,face generation-related tasks)

* AVA-ActiveSpeaker (Speech-related task, speaker-related task)

* Kinetics-400 (Action recognition)

* EPIC-KITCHENS (Action recognition)

* CMU-MOSI (Emotion recognition)

* CMU-MOSEI (Emotion recognition)

* VGGSound (Action recognition, sound localization)

* AudioSet (Action recognition, sound sepearation)

* Greatest Hits ()Sound generation

* MUSIC (Sound seperation, sound localization)

* FAIR-Play (Spatial sound generation)

* YT-ALL (Spatial sound generation)

* Replica (Depth estimation)

* AIST++ (Dance generation)

* TED (Gesture generation)

* SumMe (Saliency detection)

* AVE (Event localization)

* LLP (Event parsing)

* SoundSpaces (Audio-visual navigation)

* AVSD (Audio-visual dialog)

* Pano-AVQA (Audio-visual question answering)

* MUSIC-AVQA (Audio-visual question answering)

* AVSBench (Audio-visual segmentation, sound localization)

* HDTF (Face generation)

* MEAD (Face generation)

* RAVDESS (Face generation)

* GRID (Face generation)

* TCD-TIMIT (Speech recognition)

* CN-CVS (Continuous Visual to Speech Synthesis)

* SoundNet (sound representation from unlabeled video)

* ACAV100M (large-scale data of high audio-visual)

* SEWA-DB (emotion  and  sentiment  research in the wild)

## Review and survey

* [Image Vis. Comput.-2014] A Review of Recent Advances in Visual Speech Decoding

* [2015] Audiovisual Fusion: Challenges and New Approaches

* [2017] Multimedia Datasets for Anomaly Detection: A Review

* [2017] Multimodal Machine Learning: A Survey and Taxonomy

* [2018] A Survey of Multi-View Representation Learning

* [Int. J. Adv. Robot. Syst-2020] Audiovisual Speech Recognition: A Review and Forecast

* [2021] A Survey on Audio Synthesis and Audio-Visual Multimodal Processing

* [2021] An Overview of Deep-Learning-Based Audio-Visual Speech Enhancement and Separation

* [2021] Deep Audio-visual Learning: A Survey

* [2022] Learning in Audio-visual Context: A Review, Analysis, and New Perspective

* [2022] A review of deep learning techniques in audio event recognition (AER) applications

* [2022] Audio self-supervised learning: A survey

* [2022] Deep Learning for Visual Speech Analysis: A Survey

* [2022] Recent Advances and Challenges in Deep Audio-Visual Correlation Learning

* [2023] A Review of Recent Advances on Deep Learning Methods for Audio-Visual Speech Recognition


## Diffusion model

* [CVPR-2023] Taming Diffusion Models for Audio-Driven Co-Speech Gesture Generation

* [CVPR-2023] MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation

* [CVPR-2023] Physics-Driven Diffusion Models for Impact Sound Synthesis from Videos

* [ICCV-2023] DiffV2S: Diffusion-based Video-to-Speech Synthesis with Vision-guided Speaker Embedding

* [ICASSP-2023] Imaginary Voice: Face-styled Diffusion Model for Text-to-Speech


## Citation
#### If you find this repository helpful for your work, please kindly cite:

```bibtex
@misc{yang2023Audiovisionmodal,
  title={Awesome-AudioVision-Multimodal},
  author={Yang, Yiyuan},
  journal = {GitHub repository},
  year={2023}
}
```
