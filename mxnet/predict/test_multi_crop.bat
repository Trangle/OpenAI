@echo off
:: H:\data\bot_animal\data multi_vali.lst 列表生成
echo =======================================================
echo generate multi_vali.lst files in folder H:\data\bot_animal\data
echo =======================================================
echo.&echo.
python gen_img_list.py ^
--image-folder=data/vali/ ^
--out-folder=data/vali/ ^
--out-file=multi_vali.lst ^
--train ^
--percent-val=0.0
echo =======================================================
echo generate multi_vali.lst end...
echo =======================================================
echo.&echo.

@echo off
:: H:\data\bot_animal\data multi_vali.rec 图片转换
echo =======================================================
echo transform multi_vali.lst files in folder H:\data\bot_animal\data
echo =======================================================
echo.&echo.
python multi_im2rec.py ^
multi_vali ^
H:\data\bot_animal\data ^
--saving-folder=H:\data\bot_animal\data\vali ^
--num-thread=12 ^
--multi-crop=True ^
--short-edges=224,256,320,384,480,640 ^
--resize=224 ^
--quality=90
echo =======================================================
echo transform multi_vali.rec end...
echo =======================================================
echo.&echo.