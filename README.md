# car-number-detection

Build docker container:

```
sudo docker build .
```

Enter it:

```
sudo docker run -i -t YOUR_ID /bin/bash
```

Train the model:

```
python3 -m cnd.ocr.train_script -en DIR_NAME_YOU_WANT
```

Run on the video:

```
python3 -m worker.run 
--log_path worker/YOUR_LOG_FILENAME.txt 
--level INFO 
--video_path videos/INPUT_VIDEO_NAME.MOV 
--save_path videos/OUTPUT_NAME.MOV 
--model_path cnd/exp/exp2/model-250-0.020266.pth 
--car_number Y726HK163
```
