PS C:\Users\VINH\Downloads\YOLOv5-DEMO\YOLOv5-DEMO\yolov5> Set-Location "c:/Users/VINH/Downloads/YOLOv5-DEMO/YOLOv5-DEMO/yolov5"; Get-ChildItem runs/demo/plate_demo_1to6 -File | Select-Object -ExpandProperty Name

Demo bat luu anh bien so da cat ra:
PS> Set-Location "c:/COURSES/XLTGMA/XuLyAnh/YOLOv5-DEMO/YOLOv5-DEMO/yolov5"
PS> $env:ENABLE_PLATE_CROP='1'
PS> $env:PLATE_CROP_DIR='runs/demo/plate_crops'
PS> $env:IMAGE_NAME='test3.jpg'
PS> $env:MODEL_PATH='runs/train/ket_qua_3202/weights/best.pt'
PS> $env:OUTPUT_NAME='runs/demo/ket_qua_cuoi_cung.jpg'
PS> python run_plate.py

Kiem tra file da tao:
PS> Get-ChildItem runs/demo/plate_crops -File | Select-Object Name,Length,LastWriteTime
PS> Test-Path runs/demo/ket_qua_cuoi_cung.jpg

Demo he thong web:
PS> Set-Location "c:/COURSES/XLTGMA/XuLyAnh/YOLOv5-DEMO/YOLOv5-DEMO/yolov5/webapp"
PS> python app.py

Mo trinh duyet:
http://127.0.0.1:5000

Test nhanh API bang test1 -> test5:
PS> Set-Location "c:/COURSES/XLTGMA/XuLyAnh/YOLOv5-DEMO/YOLOv5-DEMO"
PS> foreach ($n in 1..5) {
>>   $env:N = "$n"
>>   python -c "import os,requests;from pathlib import Path;n=int(os.environ['N']);p=Path('yolov5')/f'test{n}.jpg';files={'image':(p.name,p.read_bytes(),'image/jpeg')};r=requests.post('http://127.0.0.1:5000/api/recognize',files=files,timeout=120);print('test',n,r.status_code,r.json().get('count'))"
>> }

Thu muc ket qua web:
PS> Get-ChildItem "c:/COURSES/XLTGMA/XuLyAnh/YOLOv5-DEMO/YOLOv5-DEMO/yolov5/webapp/static/results" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 5 Name,LastWriteTime