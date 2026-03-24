Assignments for [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).

See also https://github.com/ccmoony/UCB-CS285

```bash
for i in {1..5}; do
    mkdir -p "hw$i"
    micromamba env export -n "cs285_hw$i" > "hw$i/conda_environment.yml"
done
```