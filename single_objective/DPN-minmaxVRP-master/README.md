# Solving NP-hard Min-max Routing Problems as Sequential Generation with Equity Context


## Training

To train the model on MTSP, MPDP, and (F)MDVRP instances with 50 nodes and a variable number of agents ranging from 2 to 10:

```bash
python run_tsp.py --graph_size 50 --problem mtsp --run_name 'mtsp50' --agent_min 2 --agent_max 10
```

```bash
python run_pdp.py --graph_size 50 --problem mpdp --run_name 'mpdp50' --agent_min 2 --agent_max 10
```

```bash
python run_md.py --graph_size 50 --problem mdvrp --run_name 'mdvrp50' --agent_min 2 --agent_max 10 --depot_min 3 --depot_max 7
```

```bash
python run_md.py --graph_size 50 --problem fdvrp --run_name 'fmdvrp50' --agent_min 2 --agent_max 10 --depot_min 3 --depot_max 7
```



```bash
python run_tsp.py --graph_size 100 --problem mtsp --run_name 'mtsp100' --agent_min 2 --agent_max 10
```

```bash
python run_pdp.py --graph_size 100 --problem mpdp --run_name 'mpdp100' --agent_min 2 --agent_max 10
```

```bash
python run_md.py --graph_size 100 --problem mdvrp --run_name 'mdvrp100' --agent_min 2 --agent_max 10 --depot_min 5 --depot_max 10
```

```bash
python run_md.py --graph_size 100 --problem fdvrp --run_name 'fmdvrp100' --agent_min 2 --agent_max 10 --depot_min 5 --depot_max 10
```



### Pre-trained Model

https://drive.google.com/file/d/1UWs7ajI3gHYADH5RGZn8WDKto_N46yC7/view?usp=drive_link
