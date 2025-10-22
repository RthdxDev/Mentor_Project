## Experimental Results

### Full Rank Matrices

| Matrix Size   | Density | Avg Iterations | Avg Time (sec) |
|---------------|---------|----------------|----------------|
| 500x500       | 0.3     | 1.2            | 0.142          |
| 1000x1000     | 0.3     | 1.3            | 0.699          |

### Singular Matrices with RHS in Image

#### Rank ≤ 1000

| Density | Avg Iterations | Avg Time (sec) |
|---------|----------------|----------------|
| 0.5     | 1.6            | 3.224          |
| 0.3     | 1.3            | 3.163          |
| 0.1     | 1.1            | 3.121          |

#### Rank ≤ 500

| Density | Avg Iterations | Avg Time (sec) |
|---------|----------------|----------------|
| 0.5     | 1.7          | 4.578         |
| 0.3     | 1.4            | 4.341         |
| 0.1     | 1.5            | 3.574          |

#### Rank ≤ 100

| Density | Avg Iterations | Avg Time (sec) |
|---------|----------------|----------------|
| 0.5     | 1.3            | 4.439          |
| 0.3     | 1.4            | 3.065          |
| 0.1     | 1.0           | 1.019          |


**Note**: All experiments were run on 10 random systems (epochs=10).