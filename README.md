🔷 プロジェクト名

Color-Based Object Detection and Robotic Manipulation System

🔷 概要

本プロジェクトでは、PythonとOpenCVを用いて、カメラ映像から対象物の色・位置・姿勢を推定し、ロボットアームによる把持・搬送を行うシステムを開発した。

<img width="640" height="480" alt="Pick Camera_screenshot_16 01 2026" src="https://github.com/user-attachments/assets/22c2b92f-ad31-4c51-b62e-4a0362dade6d" />

俯瞰視点にカメラを移動させた後、画像処理により対象物（ペン）の位置と傾きを算出し、把持姿勢を補正して掴み動作を実現する。

<img width="640" height="480" alt="Pick Camera_screenshot_13 01 2026" src="https://github.com/user-attachments/assets/93bdd448-5bdb-4a09-b469-bcfc6307b554" />

🔷 Features
- Color-based object detection (HSV)
- Object position estimation (centroid)
- Orientation estimation using PCA
- Robotic manipulation (hardware-dependent)

🔷 システム構成

カメラ（エンドエフェクタに固定）

Python

OpenCV

ロボットアーム

マーカー認識によるゴール判定

🔷 処理フロー

ロボットアームを上昇させ俯瞰視点を取得

カメラ映像を取得

HSV空間で色検出

輪郭抽出 → 重心計算

主成分分析（PCA）による傾き推定

把持角度を補正して対象物を掴む

障害物を回避しながらゴールへ移動

マーカー認識した箱へ収納

🔷 技術的ポイント

OpenCVによるリアルタイム画像処理

<img width="320" height="240" alt="nav_screenshot_16 01 2026" src="https://github.com/user-attachments/assets/871987ff-c1a3-4119-b0a7-634cf2b7312c" />

PCAを用いた物体姿勢推定

センサ情報に基づく制御アルゴリズム

<img width="300" height="400" alt="Map_screenshot_21 01 2026" src="https://github.com/user-attachments/assets/e107d677-67c0-44ad-ab80-f4b2e92634fc" />

認識から制御までの統合システム構築

🔷 成果

個別把持成功率：約92.6%（ペン対象）

タスク完了率：約80%

🔷 課題

カメラの振動による認識精度低下
![IMG_2026-01-16-18-44-33-686](https://github.com/user-attachments/assets/cfb11381-6566-4f2d-b4f8-2848e7163ca1)

センサ誤差による把持ズレ

🔷 今後の改善

カメラの固定による把持精度向上

深層学習による物体認識への拡張

🔷 デモ動画

https://youtu.be/XHCQg8P0yd4
(PC画面)

https://youtu.be/hbV2CIgUGOc
(動作の様子)


🔷 実行方法
python main.py

🔷 Project Structure
src/
  main.py
demo/
  動画

🔷 Tech Stack
- Python
- OpenCV
- NumPy

🔷 Environment
- Dobot Magician
- CyberPi (mbot2)
- Raspberry Pi 4B
- micro SD card 16GB
- web camera
- cariblation sheet

※DobotおよびCyberPi関連ライブラリは公式SDKが必要です

※一部モジュールはローカル環境用のため省略しています

🔷 Note
This project includes hardware-dependent modules (Dobot, CyberPi, custom modules).

Due to environment differences, some parts of the code may not run without the actual hardware setup.

However, the core logic for image processing (color detection, position estimation, PCA-based orientation estimation) is fully implemented in Python using OpenCV.



