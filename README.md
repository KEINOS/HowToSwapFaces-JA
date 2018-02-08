[この記事](https://blog.keinos.com/20180207_3332)は、ディープラーニング（Tensorflow と OpenCV）を使った**動画内の顔の置き換えのチュートリアル**（Mac版）です。

必ず、以下を読み、理解した上で本記事をお読みください。

- <del>注意記事「[What I've learned](https://www.reddit.com/r/deepfakes/comments/7tjjbv/what_ive_learned/)」</del> @ reddit
- <del>参考記事「[Tutorial for Mac ](https://www.reddit.com/r/deepfakes/comments/7ttqk2/tutorial_for_mac/)」</del> @ reddit
- [Deepfakes の作り方から学ぶ機械学習と SNS などの画像公開の危険性](https://blog.keinos.com/20180207_3285)

2018/02/08: 参考記事の著者のアカウントが閉じられたため、リンクが切れました。主な内容としては、「偽装ポルノに使うな」です。ここでも同じことを強く言います。

## はじめに

この記事は人物 A の顔を、人物 B の動画の顔に置き換える技術文書です。流れを見ればわかりますが、特別なことをしておらず、すでに[既存の公開されている技術を組み合わせているだけ](https://blog.keinos.com/20180207_3285#outline__2)だとわかると思います。

この記事の目的は、アタリで描かれたアニメーションや漫画に、機械学習したキャラクターの顔を後から置き換えることで、**アニメーターや漫画家の負荷を減らすことは出来ないかという前衛的な実験**を目的として作成されています。原文は [GitHub](https://github.com/KEINOS/HowToSwapFaces-JA) に起きますので、何か活用案やアイデアなどありましたら [issue を立てて](https://github.com/KEINOS/HowToSwapFaces-JA/issues)ください。なお、冷やかしなど不本意な issue はリポジトリのオーナー権限と独断で削除いたします。

## Deepfakes のチュートリアル

### 下準備

1. CPU で処理するのか GPU で処理するか決める

2. FaceSwap アプリをダウンロード＆解凍する
	- https://github.com/deepfakes/faceswap/archive/master.zip
	- 解凍フォルダを "faceswap" に変更する

3. "faceswap/"内に以下の 6 つのフォルダを作成してわかりやすい名前にする。
	1. 人物 A の素材（例："Me/"など）
	2. 人物 A の顔（例："My_face/"など）
	3. 人物 B の素材（例："He/"など）
	4. 人物 B の顔（例："His_face/"など）
	5. モデル置き場（例："model/"）
	6. 出力先（例："output/"）

4. 「人物 B の素材」に動画を置く

5. 「人物 A の素材」にありったけの画像を置く

### 学習に必要なソフトウェア類やモデルのインストール

6. NVIDIA のサイトから Mac用 CUDA ドライバをダウンロードする。
	- http://www.nvidia.co.jp/object/mac-driver-archive-jp.html
	- 十分な GPU がないといったメッセージが出ても、気にせずインストールしてしまいましょう。

7. 学習済みモデルのコピー
	- Windows用 FakeApp に同梱されている学習モデルをコピーします
	- https://mega.nz/#!hTgA2b6b!mI6k9dFt_w__jIEUQO2ZePhzFMg6JWUpBZWiV2TDgs4 
	- 上記リンクから "FakeApp.zip" をダウンロードし、以下のファイル（学習済みモデル）を「モデル置き場」にコピーします。
		- `encoder.h5`
		- `decoder_A.h5`
		- `decoder_B.h5`
	- 残りはいらないので ZIP ファイル含め削除します。後に学習済みモデルが壊れた場合を考えてバックアップをしておくといいかもしれません。
	- 以後は、この学習済みモデルを叩き上げていくことになります。（独自モデルを使っても可能）

8. Anaconda が入っていない場合はダウンロードしてインストールします。（どちらのバージョンでも構いません）
	- Anaconda 2.7
		- https://repo.continuum.io/archive/Anaconda2-5.0.1-MacOSX-x86_64.pkg
	- Anaconda 3.6
		- https://repo.continuum.io/archive/Anaconda3-5.0.1-MacOSX-x86_64.pkg
9. NVIDIA cuDNN をダウンロード＆解凍します。
	- https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/cudnn-8.0-osx-x64-v6.0-tgz
	- ユーザー登録が必要です。

10. 解凍したファイルを以下のフォルダにそれぞれコピーします。
    - `/Developer/NVIDIA/CUDA-8.0/lib`
    - `/Developer/NVIDIA/CUDA-8.0/include`
    - 上記フォルダがない場合は作成します。CUDA ドライバによってバージョン番号が違うかもしれません。

11. ターミナルを開き、"faceswap" フォルダにいる（カレントディレクトリである）ことを確認してください。
	- `$ cd /path/to/your/faceswap/`

12. [Python の仮想環境](https://qiita.com/caad1229/items/325ca5c8ad198b0ebce7)を作る
	- `$ pip install virtualenv` 
	- インストールに失敗する場合は、下記依存ファイルを先にインストールしてみてから再度試してください。

13. 仮想環境に依存ファイルをインストールする
	- CPU を使う場合
		- `$ pip install -r requirements.txt`
	- GPU を使う場合
		- `$ pip install -r requirements-gpu.txt`

14. FFmpeg が入っていない場合は[インストール](https://qiita.com/search?utf8=%E2%9C%93&sort=&q=Mac+FFMpeg+%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB)します
	- `$ ffmpeg` でバージョンが表示されるならOK

### 学習素材の下ごしらえ①（動画から静止画を作成）

15. 「人物 B の素材」ディレクトリに移動します
	- `$ cd /path/to/your/faceswap/[人物 B の素材]/`
	- 動画があることを確認し、ファイル名を英数字スペースなしに変更します。（以下は`scene.mp4`に変更する場合。Finderで変更も可）
		- `mv ./[動画名].mp4 ./`

16. [動画を静止画に展開](https://qiita.com/matoken/items/664e7a7e8f31e8a46a60)
	- `$ ffmpeg -i scene.mp4 -vf fps=30 scene%06d.png`

### 学習環境の動作チェック

17. 親ディレクトリ（`/path/to/your/faceswap/`）に戻る
    - `$ cd ../`

18. 仮想環境を立ち上げる
	- `virtualenv faceswap_env/`

19. 仮想環境が立ち上がったら以下のヘルプを実行して環境チェックを行う
	- `python faceswap.py -h`
	- ヘルプが表示され、"Tensorflow Backend"的なメッセージが出て入ればOK
	- 上記のメッセージが出ない場合は、以下の導入に問題がある可能性があります。
		- `Cmake`
		- `OpenCV`
		- [`Dlib`](https://qiita.com/nonbiri15/items/9561c8194ba0b2041bd0) 
	- "cv2 import..."のようなメッセージの場合
		- `$ pip install opencv-python`
	- "dlib..."のようなメッセージの場合
		- `$ conda install dlib` もしくは
		- `$ pip install dlib`
	- 上記でダメな場合は
		- `$ conda install cmake` もしくは
		- `$ pip install cmake`
	- さらに上記でもダメな場合は、以下をダウンロード＆解凍したディレクトリに移動後、インストールを試してみてください
		- http://dlib.net/files/dlib-19.9.tar.bz2
		- `$ cd /path/to/your/donwloaded/extracted/dlib`
		- `$ python setup.py install`

### 学習素材の下ごしらえ②（画像から顔のみを抽出）

20. 人物 B の動画の展開画像（静止画）から顔を抽出
	- `cd /path/to/your/faceswap`
	- `python faceswap.py extract -i 「人物 B の素材」のパス -o 「人物 B の顔」のパス`
	- ここで抽出した画像をチェックし、NG画像を削除すると精度があがります。

21. 人物 A の画像から顔を抽出
	- `cd /path/to/your/faceswap`
	- `python faceswap.py extract -i 「人物 A の素材」のパス -o 「人物 A の顔」のパス`
	- ここで抽出した画像をチェックし、NG画像を削除すると精度があがります。

### 機械学習の実行

22. それぞれの顔の特徴を学習させる
	- `$ python faceswap.py train -A 「人物 B の顔」のパス -B 「人物 A の顔」のパス -m 「モデル置き場」 -p`

23. 学習が終わったら`q`をタイプして学習内容を保存させる。

### 顔の入れ替えと静止画からの動画作成

24. 画像の顔の入れ替えを実行する
	- `python faceswap.py convert -i 「人物 B の素材」のパス -o 「出力先」のパス -m「モデル置き場」`

25. 置き換えた画像を動画に変換する
	- `ffmpeg -i scene%06d.png -c:v libx264 -vf "fps=30,format=yuv420p" output.mp4`


