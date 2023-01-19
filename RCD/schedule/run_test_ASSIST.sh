cd ..
cp config_ASSIST.txt config.txt
cp ../data/ASSIST/log_data_ori.json ../data/ASSIST/log_data.json
cp ../data/ASSIST/train_set_ori.json ../data/ASSIST/train_set.json
cp ../data/ASSIST/valid_set_ori.json ../data/ASSIST/valid_set.json
cp ../data/ASSIST/test_set_ori.json ../data/ASSIST/test_set.json

cp ../data/ASSIST/graph/K_Directed_ori.txt ../data/ASSIST/graph/K_Directed.txt
cp ../data/ASSIST/graph/K_Undirected_ori.txt ../data/ASSIST/graph/K_Undirected.txt

python build_k_e_graph.py --dir ../data/ASSIST/
python build_u_e_graph.py --dir ../data/ASSIST/

nohup python -u test.py --dir ../data/ASSIST/ --gpu 0 --mode 0 --log ASSIST_full_1 > /dev/null &
nohup python -u test.py --dir ../data/ASSIST/ --gpu 1 --mode 0 --log ASSIST_full_2 > /dev/null &
nohup python -u test.py --dir ../data/ASSIST/ --gpu 2 --mode 0 --log ASSIST_full_3 > /dev/null &
sleep 5

cp ../data/ASSIST/graph/K_Directed_no.txt ../data/ASSIST/graph/K_Directed.txt
cp ../data/ASSIST/graph/K_Undirected_no.txt ../data/ASSIST/graph/K_Undirected.txt

python build_k_e_graph.py --dir ../data/ASSIST/
python build_u_e_graph.py --dir ../data/ASSIST/

nohup python -u test.py --dir ../data/ASSIST/ --gpu 0 --mode 0 --log ASSIST_no_dep_1 > /dev/null &
nohup python -u test.py --dir ../data/ASSIST/ --gpu 1 --mode 0 --log ASSIST_no_dep_2 > /dev/null &
nohup python -u test.py --dir ../data/ASSIST/ --gpu 2 --mode 0 --log ASSIST_no_dep_3 > /dev/null &
sleep 5

cp ../data/ASSIST/graph/K_Directed_noise.txt ../data/ASSIST/graph/K_Directed.txt
cp ../data/ASSIST/graph/K_Undirected_noise.txt ../data/ASSIST/graph/K_Undirected.txt

python build_k_e_graph.py --dir ../data/ASSIST/
python build_u_e_graph.py --dir ../data/ASSIST/

nohup python -u test.py --dir ../data/ASSIST/ --gpu 0 --mode 0 --log ASSIST_noise_1 > /dev/null &
nohup python -u test.py --dir ../data/ASSIST/ --gpu 1 --mode 0 --log ASSIST_noise_2 > /dev/null &
nohup python -u test.py --dir ../data/ASSIST/ --gpu 2 --mode 0 --log ASSIST_noise_3 > /dev/null &
sleep 5

cp config_ASSIST_0.txt config.txt
cp ../data/ASSIST/log_data_0.json ../data/ASSIST/log_data.json
cp ../data/ASSIST/train_set_0.json ../data/ASSIST/train_set.json
cp ../data/ASSIST/valid_set_0.json ../data/ASSIST/valid_set.json
cp ../data/ASSIST/test_set_0.json ../data/ASSIST/test_set.json

cp ../data/ASSIST/graph/K_Directed_no.txt ../data/ASSIST/graph/K_Directed.txt
cp ../data/ASSIST/graph/K_Undirected_no.txt ../data/ASSIST/graph/K_Undirected.txt

python build_k_e_graph.py --dir ../data/ASSIST/
python build_u_e_graph.py --dir ../data/ASSIST/

nohup python -u test.py --dir ../data/ASSIST/ --gpu 0 --mode 0 --log ASSIST_no_con_1 > /dev/null &
nohup python -u test.py --dir ../data/ASSIST/ --gpu 1 --mode 0 --log ASSIST_no_con_2 > /dev/null &
nohup python -u test.py --dir ../data/ASSIST/ --gpu 2 --mode 0 --log ASSIST_no_con_3 > /dev/null &
sleep 5