#Train IGANet
python main.py --train --model model_IGANet --layers 3 --nepoch 20 --gpu 0
#Test IGANet
# python main.py --reload --previous_dir "./pre_trained_model" --model model_IGANet --layers 3 --gpu 1