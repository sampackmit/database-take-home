
## Solution 

### Approach & Analysis

[Describe how you analyzed the query patterns and what insights you found]

I saw that there was an exponential decay in the queries, so it is most important to make sure that a small subset of the nodes are reachable quickly. This led to a strategy focused on only the most likely nodes and ignoring the unlikely ones.  

### Optimization Strategy

[Explain your optimization strategy in detail]

My graph roughly has two sets of nodes. One is a cycle that goes through each of the nodes, and the other is a set of edges that go from each node to the highest query frequency nodes. The top two nodes are chosen by via the nodes with the top two weights for the outdegrees of the edges.  

### Implementation Details

[Describe the key aspects of your implementation]

My implementation makes essentially an entirely new graph. It identifies the nodes with the highest query frequencies and adds edges going to those nodes. It additionally adds a random cycle between all of the nodes. 

### Results

[Share the performance metrics of your solution]

SUCCESS RATE:
  Initial:   79.5% (159/200)
  Optimized: 100.0% (200/200)
  ✅ Improvement: 20.5%

PATH LENGTHS (successful queries only):
  Initial:   545.0 (159/200 queries)
  Optimized: 123.5 (200/200 queries)
  ✅ Improvement: 77.3%

COMBINED SCORE (success rate × path efficiency):
  Score: 268.88

### Trade-offs & Limitations

[Discuss any trade-offs or limitations of your approach]

My approach does better at high success rate than path length. This is because of the cycle structure it uses. 

### Iteration Journey

[Briefly describe your iteration process - what approaches you tried, what you learned, and how your solution evolved]

I initially started with a strategy of focusing on going directly to the highest frequency nodes, then I tried to adjust that by adding cycles with small probability. However, I found that the cycles worked better overall, and I went with that. 

---

* Be concise but thorough - aim for 500-1000 words total
* Include specific data and metrics where relevant
* Explain your reasoning, not just what you did