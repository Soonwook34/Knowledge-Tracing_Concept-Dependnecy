cd ..
cp config_KSC2022.txt config.txt
cp ../data/gendata/log_data_KSC.json ../data/gendata/log_data.json
cp ../data/gendata/train_set_KSC.json ../data/gendata/train_set.json
cp ../data/gendata/valid_set_KSC.json ../data/gendata/valid_set.json
cp ../data/gendata/test_set_KSC.json ../data/gendata/test_set.json

cp ../data/gendata/graph/K_Directed_2_6.txt ../data/gendata/graph/K_Directed.txt
cp ../data/gendata/graph/K_Undirected_2_6.txt ../data/gendata/graph/K_Undirected.txt

python build_k_e_graph.py --dir ../data/gendata/
python build_u_e_graph.py --dir ../data/gendata/

nohup python -u main.py --student_n 100 --exer_n 300 --knowledge_n 6 --dir ../data/gendata/ --gpu 0 --epoch_n 200 --lr 0.005 --mode 0 --log KSC_full_1 > /dev/null &
nohup python -u main.py --student_n 100 --exer_n 300 --knowledge_n 6 --dir ../data/gendata/ --gpu 1 --epoch_n 200 --lr 0.005 --mode 0 --log KSC_full_2 > /dev/null &
nohup python -u main.py --student_n 100 --exer_n 300 --knowledge_n 6 --dir ../data/gendata/ --gpu 2 --epoch_n 200 --lr 0.005 --mode 0 --log KSC_full_3 > /dev/null &
sleep 10

cp ../data/gendata/graph/K_Directed_2_6_no.txt ../data/gendata/graph/K_Directed.txt
cp ../data/gendata/graph/K_Undirected_2_6_no.txt ../data/gendata/graph/K_Undirected.txt

python build_k_e_graph.py --dir ../data/gendata/
python build_u_e_graph.py --dir ../data/gendata/

nohup python -u main.py --student_n 100 --exer_n 300 --knowledge_n 6 --dir ../data/gendata/ --gpu 0 --epoch_n 200 --lr 0.005 --mode 0 --log KSC_no_dep_1 > /dev/null &
nohup python -u main.py --student_n 100 --exer_n 300 --knowledge_n 6 --dir ../data/gendata/ --gpu 1 --epoch_n 200 --lr 0.005 --mode 0 --log KSC_no_dep_2 > /dev/null &
nohup python -u main.py --student_n 100 --exer_n 300 --knowledge_n 6 --dir ../data/gendata/ --gpu 2 --epoch_n 200 --lr 0.005 --mode 0 --log KSC_no_dep_3 > /dev/null &
sleep 10

cp config_KSC2022_0.txt config.txt
cp ../data/gendata/log_data_KSC_0.json ../data/gendata/log_data.json
cp ../data/gendata/train_set_KSC_0.json ../data/gendata/train_set.json
cp ../data/gendata/valid_set_KSC_0.json ../data/gendata/valid_set.json
cp ../data/gendata/test_set_KSC_0.json ../data/gendata/test_set.json

cp ../data/gendata/graph/K_Directed_2_6_0.txt ../data/gendata/graph/K_Directed.txt
cp ../data/gendata/graph/K_Undirected_2_6_0.txt ../data/gendata/graph/K_Undirected.txt

python build_k_e_graph.py --dir ../data/gendata/
python build_u_e_graph.py --dir ../data/gendata/

nohup python -u main.py --student_n 100 --exer_n 300 --knowledge_n 1 --dir ../data/gendata/ --gpu 0 --epoch_n 200 --lr 0.005 --mode 0 --log KSC_no_con_1 > /dev/null &
nohup python -u main.py --student_n 100 --exer_n 300 --knowledge_n 1 --dir ../data/gendata/ --gpu 1 --epoch_n 200 --lr 0.005 --mode 0 --log KSC_no_con_2 > /dev/null &
nohup python -u main.py --student_n 100 --exer_n 300 --knowledge_n 1 --dir ../data/gendata/ --gpu 2 --epoch_n 200 --lr 0.005 --mode 0 --log KSC_no_con_3 > /dev/null &
sleep 10

cp config_KSC2022.txt config.txt
cp ../data/gendata/log_data_KSC.json ../data/gendata/log_data.json
cp ../data/gendata/train_set_KSC.json ../data/gendata/train_set.json
cp ../data/gendata/valid_set_KSC.json ../data/gendata/valid_set.json
cp ../data/gendata/test_set_KSC.json ../data/gendata/test_set.json

cp ../data/gendata/graph/K_Directed_2_6_noise.txt ../data/gendata/graph/K_Directed.txt
cp ../data/gendata/graph/K_Undirected_2_6_noise.txt ../data/gendata/graph/K_Undirected.txt

python build_k_e_graph.py --dir ../data/gendata/
python build_u_e_graph.py --dir ../data/gendata/

nohup python -u main.py --student_n 100 --exer_n 300 --knowledge_n 6 --dir ../data/gendata/ --gpu 0 --epoch_n 200 --lr 0.005 --mode 0 --log KSC_noise_1 > /dev/null &
nohup python -u main.py --student_n 100 --exer_n 300 --knowledge_n 6 --dir ../data/gendata/ --gpu 1 --epoch_n 200 --lr 0.005 --mode 0 --log KSC_noise_2 > /dev/null &
nohup python -u main.py --student_n 100 --exer_n 300 --knowledge_n 6 --dir ../data/gendata/ --gpu 2 --epoch_n 200 --lr 0.005 --mode 0 --log KSC_noise_3 > /dev/null &
sleep 10