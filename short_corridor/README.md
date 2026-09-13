## Short Corridor With Switched Actions

We are given a corridor in the form of grid. There are four cells in this grid.
```
┌────────────────┬─────────────────┬─────────────────┬─────────────────────┐
│                │                 │                 │                     │
│      S         │                 │                 │          T          │
└────────────────┴─────────────────┴─────────────────┴─────────────────────┘
```
The reward is 1 per step, as usual. In each of the three nonterminal states there are only two actions, right and left. 

These actions have their usual consequences in the first and third states (left causes no movement in the first state), but in the second state they are reversed, so that right moves to the left and left moves to the right. The problem is difficult because all the states appear identical under the function approximation. In particular, we define x(s,right) = [1, 0] and
x(s, left) = [0, 1], for all s. 

An action-value method with E-greedy action selection is forced to choose between just two policies: choosing right with high probability 1 - E/2 on all steps or choosing left with the same high probability on all time steps. If E =0.1, then these two policies achieve a value (at the start state)
of less than 44 and 82, respectively. A method can do significantly better if it can learn a specific probability with which to select right. The best probability is about 0.59, which achieves a value of about 11.6.