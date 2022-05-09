#python context_dim_experiment.py 5 &
#python context_dim_experiment.py 10 &
#wait
#python context_dim_experiment.py 15 &
#python context_dim_experiment.py 20 &
#wait
#python context_dim_experiment.py 25 &
#python context_dim_experiment.py 30 &
#wait
python domain_shift_simulation.py 0 &
python domain_shift_simulation.py 1 &
wait
python domain_shift_simulation.py 2 &
python domain_shift_simulation.py 3 &
wait
python domain_shift_simulation.py 4 &
python domain_shift_simulation.py 5 &
wait
python domain_shift_simulation.py 6 &
python domain_shift_simulation.py 7 &
wait
python domain_shift_simulation.py 8 &
python domain_shift_simulation.py 9 &
wait
#python item_size_experiment.py 20 &
#python item_size_experiment.py 40 &
#wait
#python item_size_experiment.py 60 &
#wait
#python item_size_experiment.py 80 &
#wait
#python user_size_experiment.py 50 &
#python user_size_experiment.py 100 &
#wait
#python user_size_experiment.py 150 &
#python user_size_experiment.py 200 &
#wait
#python user_size_experiment.py 250 &
#python user_size_experiment.py 300 &
#wait
python recommendation_size_experiment.py 2 &
python recommendation_size_experiment.py 3 &
wait
python recommendation_size_experiment.py 4 &
python recommendation_size_experiment.py 5 &
wait
python recommendation_size_experiment.py 6 &
python recommendation_size_experiment.py 7 &
wait
#python sample_size_experiment.py 100 &
#python sample_size_experiment.py 500 &
#wait
#python sample_size_experiment.py 1000 &
#python sample_size_experiment.py 2500 &
#wait
#python sample_size_experiment.py 5000 &
