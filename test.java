public static void dfs(ArrayList<ArrayList<Integer>> graph, boolean[] visited, int v)
{
	visited[v] = true;
	for(Integer next : graph.get(v))
		if(!visited[next])
			dfs(graph, visited, next);
}

public static int[] bfs(ArrayList<ArrayList<Integer>> graph, int v)
{
	int n = graph.size();
	boolean[] visited = new boolean[n];
	int[] distance = new int[n];
	Arrays.fill(distance, -1);
	ArrayDeque<Integer> q = new ArrayDeque<>();
	visited[v] = true;
	q.offer(v);
	while(q.size() > 0)
	{
		int cur = q.poll();
		for(Integer next : graph.get(cur))
		{
			if(distance[next] == -1)
			{
				distance[next] = distance[cur]+1;
				q.offer(next);
			}
		}
	}
	return distance;
}

public static ArrayDeque<Integer> toposort(ArrayList<ArrayList<Integer>> graph, int[] incoming)
{
	ArrayDeque<Integer> q = new ArrayDeque<>();
	ArrayDeque<Integer> sort = new ArrayDeque<>();
	int total = incoming.length;
	for(int i = 0; i < incoming.length; i++)
		if(incoming[i] == 0)
			q.offer(i);
	while(!q.isEmpty())
	{
		int at = q.poll();
		sort.offer(at);
		total--;
		for(int edge : graph.get(at))
		{
			incoming[edge]--;
			if(incoming[edge] == 0)
				q.offer(edge);
		}
	}
	return total == 0 ? sort : null;
}

// Edge {v1, v2, w} -> graph[v1][v2] = w
public static ArrayList<Edge> prim(ArrayList<ArrayList<Edge>> graph, int v)
{
	int n = graph.size(), numEdges = 0;
	boolean[] visited = new boolean[n];
	ArrayList<Edge> mst = new ArrayList<>();
	PriorityQueue<Edge> pq = new PriorityQueue<>();

	for(Edge e : graph.get(v))
		pq.offer(e);
	visited[v] = true;
	while(!pq.isEmpty())
	{
		Edge next = pq.poll();
		if(visited[next.v2]) continue;
		for(Edge e : graph.get(next.v2))
			if(!visited[e.v2])
				pq.offer(e);
		visited[next.v2] = true;
		mst.add(next);
		numEdges++;
		if(numEdges == n - 1) break;
	}
	return numEdges == n - 1 ? mst : null;
}

class djset
{
	public int[] parent;

	public djset(int n)
	{
		parent = new int[n];
		for (int i=0; i<n; i++)
			parent[i] = i;
	}
 
	public int find(int v)
	{
		if(parent[v] == v) return v;
		int res = find(parent[v]);
		parent[v] = res;
		return res;
	}

	public boolean union(int v1, int v2)
	{
		int rootv1 = find(v1);
		int rootv2 = find(v2);
		if(rootv1 == rootv2) return false;
		parent[rootv2] = rootv1;
		return true;
	}
}

public static ArrayList<Edge> kruskal(PriorityQueue<Edge> edges, int n)
{
	djset trees = new djset(n);
	int numEdges = 0;
	ArrayList<Edge> mst = new ArrayList<>();
	while(!edges.isEmpty())
	{
		Edge next = edges.poll();
		boolean merged = trees.union(next.v1, next.v2);
		if(!merged) continue;
		mst.add(next);
		numEdges++;
		if(numEdges == n-1) break;
	}
	return numEdges == n-1 ? mst : null;
}

// To return the shortest distance to all nodes from start: replace finish with number
// of vertices, remove the finish terminator, use dists array like Bellman-Ford
public static int dijkstras(ArrayList<ArrayList<Edge>> graph, int start, int finish)
{
	boolean[] visited = new boolean[graph.size()];
	PriorityQueue<Edge> pq = new PriorityQueue<>();
	pq.add(new Edge(start, start, 0));
	while(!pq.isEmpty())
	{
		Edge next = pq.poll();
		if(visited[next.v2]) continue;
		if(next.v2 == finish) return next.w;
		visited[next.v2] = true;
		for(Edge adj : graph.get(next.v2))
			if(!visited[adj.v2])
				pq.add(new Edge(next.v2, adj.v2, next.w + adj.w));
	}
	return -1;
}

public static int[] bellmanford(Edge[] edges, int n, int start)
{
	int oo = (int)1e9;
	int[] dists = new int[n];
	for(int i = 0; i < n; i++)
		dists[i] = oo;
	dists[start] = 0;
	for(int i = 0; i < n - 1; i++)
		for(Edge e : edges)
			if(dists[e.v1] + e.w < dists[e.v2])
				dists[e.v2] = dists[e.v1] + e.w;	
	return dists;
}

public static void floydwarhall(int[][] edges, int[][] paths, int n)
{
	int oo = (int)1e9;
	// Extension
	for(int i = 0; i < n; i++)
		for(int j = 0; j < n; j++)
			paths[i][j] = edges[i][j] < oo ? i : -1;

	for(int k = 0; k < n; k++)
		for(int i = 0; i < n; i++)
			for(int j = 0; j < n; j++)
				if (edges[i][j] > edges[i][k] + edges[k][j] && edges[i][k] < oo && edges[k][j] < oo)
				{
					edges[i][j] = edges[i][k] + edges[k][j];
					paths[i][j] = paths[k][j];
				}
}
// Extension
// if(edges[start][finish] < oo) then path exists, so reconstruct
public static ArrayDeque<Integer> getPath(int[][] paths, int start, int finish)
{
	ArrayDeque<Integer> path = new ArrayDeque<>();
	path.push(finish);
	while(finish != start)
	{
		finish = paths[start][finish];
		path.push(paths[start][finish]);
	}
	return path;
}

class NetworkFlow
{
	public int n, source, sink, oo = (int)1e9;
	public int[][] cap;

	public NetworkFlow(int size)
	{
		n = size + 2;
		source = n - 2;
		sink = n - 1;
		cap = new int[n][n];
	}

	public void add(int v1, int v2, int c)
	{
		cap[v1][v2] = c;
	}

	public int flow()
	{
		boolean[] visited = new boolean[n]; // used for dfs
		int network = 0;

		while(true)
		{
			Arrays.fill(visited, false); // used for dfs
			int res = dfs(source, visited, oo);
			// int res = bfs();
			if(res == 0) break;
			network += res;
		}

		return network;
	}

	public int dfs(int v, boolean[] visited, int min)
	{
		if(v == sink) return min;
		if(visited[v]) return 0;
		visited[v] = true;
		int network = 0;

		for(int i = 0; i < n; i++)
		{
			if(cap[v][i] > 0)
				network = dfs(i, visited, Math.min(cap[v][i], min));
			if(network > 0)
			{
				cap[v][i] -= network;
				cap[i][v] += network;
				return network;
			}
		}

		return 0;
	}

	public int bfs()
	{
		int[] reach = new int[n + 2];
		int[] prev = new int[n + 2];
		LinkedList<Integer> q = new LinkedList<Integer>();

		Arrays.fill(prev, -1);
		reach[source] = oo;
		q.offer(source);
		while (q.size() > 0)
		{
			int v = q.poll();
			if(v == sink) break;
			for(int i=0; i<n; i++)
			{
				if(prev[i] == -1 && cap[v][i] > 0)
				{
					prev[i] = v;
					reach[i] = Math.min(cap[v][i], reach[v]);
					q.offer(i);
				}
			}
		}

		if(reach[sink] == 0) return 0;
		int v1 = prev[sink];
		int v2 = sink;
		int network = reach[sink];
		while(v2 != source)
		{
			cap[v1][v2] -= network;
			cap[v2][v1] += network;
			v2 = v1;
			v1 = prev[v1];
		}

		return network;
	}
}

public static void order(Node root)
{
	print(root); // pre-order
	for(Node n : root.kids)
		if(n != null)
			order(n);
	// print(root); // post-order
}

// kD Tree
class Node
{
	int dim;
	int[] data;
	Node parent, left, right;

	public Node(int ...n)
	{
		dim = n.length;
		data = new int[dim];
		for(int i = 0; i < dim; i++)
			data[i] = n[i];
		parent = null;
		left = null;
		right = null;
	}

	public void insert(Node n, int lvl)
	{
		n.parent = this;
		if(this.compareTo(n, lvl) < 0)
			if(left == null) this.left = n;
			else this.left.insert(n, (lvl + 1) % dim);
		else
			if(right == null) this.right = n;
			else this.right.insert(n, (lvl + 1) % dim);
	}

	public Node closest(Node n, int lvl)
	{
		Node c = this;
		if(this.compareTo(n, lvl) < 0)
		{
			if(left != null)
				c = this.left.closest(n, (lvl + 1) % dim);
			if(right != null)
				if(c.dist(n) > this.right.dist(n))
					c = this.right.closest(n, (lvl + 1) % dim);
		}
		else
		{
			if(right != null)
				c = this.right.closest(n, (lvl + 1) % dim);
			if(left != null)
				if(c.dist(n) > this.left.dist(n))
					c = this.left.closest(n, (lvl + 1) % dim);
		}
		if(this.dist(n) < c.dist(n))
			c = this;
		return c;
	}

	public int dist(Node o)
	{
		int sum = 0;
		if(dim == 1) return Math.abs(this.data[0] - o.data[0]);
		for(int i = 0; i < dim; i++)
			sum += this.data[i] * this.data[i] + o.data[i] * o.data[i];
		return sum;
	}

	public int compareTo(Node o, int lvl) { return this.data[lvl] - o.data[lvl]; }
}

class SparseTable
{
	int maxk, n;
	int[][] table, idx;

	public SparseTable(int[] data)
	{
		n = data.length;
		maxk = Integer.numberOfTrailingZeros(Integer.highestOneBit(n));
		table = new int[maxk + 1][n];
		idx = new int[maxk + 1][n];

		for(int i = 0; i < n; i++)
		{
			table[0][i] = data[i];
			idx[0][i] = i;
		}
		for(int k = 1; k <= maxk; k++)
		{
			for(int i = 0; i < n; i++)
			{
				int j = i + (1 << (k - 1));
				if(j < n)
				{
					// change for type of sparse table
					if(table[k - 1][i] < table[k - 1][j])
					{
						idx[k][i] = idx[k - 1][i];
						table[k][i] = table[k - 1][i];
					}
					else
					{
						idx[k][i] = idx[k - 1][j];
						table[k][i] = table[k - 1][j];
					}
				}
			}
		}
	}

	// change according to type
	public int minimum(int a, int b)
	{
		int len = b - a + 1;
		int k = Integer.numberOfTrailingZeros(Integer.highestOneBit(len));
		int m = b - (1 << k) + 1;
		return Math.min(table[k][a], table[k][m]); // returns value that is min
		// if(table[k][a] < table[k][m]) return idx[k][a]; // returns index that is min
		// return idx[k][m];
	}
}

class BIT
{
	int[] data;

	public BIT(int n)
	{
		data = new int[2 * Integer.highestOneBit(n) + 1];
	}

	public void update(int in, int val)
	{
		while(in < data.length)
		{
			data[in] += val;
			in += in & -in;
		}
	}

	public int sum(int in)
	{
		int sum = 0;
		while(in > 0)
		{
			sum += data[in];
			in -= in & -in;
		}
		return sum;
	}

	public int rangeSum(int i, int j)
	{
		return sum(j) - sum(i - 1);
	}

	public int search(int k)
	{
		int i = data.length - 1, sum = 0;
		while(i > 0)
		{
			if(data[sum + i] <= k)
			{
				sum += i;
				k -= data[i];
			}
			i >>= 1;
		}
		return k == 0 ? sum : sum + 1;
	}
}

class SegmentTree
{
	int n;
	int[] lo, hi, data, delta;

	public SegmentTree(int n)
	{
		this.n = n;
		lo = new int[4 * Integer.highestOneBit(n - 1) + 1];
		hi = new int[4 * Integer.highestOneBit(n - 1) + 1];
		data = new int[4 * Integer.highestOneBit(n - 1) + 1];
		delta = new int[4 * Integer.highestOneBit(n - 1) + 1];
		init(1, 0, n - 1);
	}

	public void init(int i, int a, int b)
	{
		lo[i] = a;
		hi[i] = b;
		if(a == b) return;
		int m = (b - a) / 2 + a;
		init(2 * i, a, m);
		init(2 * i + 1, m + 1, b);
	}

	// lazy propagation
	public void prop(int i)
	{
		delta[2 * i] += delta[i];
		delta[2 * i + 1] += delta[i];
		delta[i] = 0;
	}

	// change for type of segment tree
	public void track(int i)
	{
		data[i] = Math.min(data[2 * i] + delta[2 * i], data[2 * i + 1] + delta[2 * i + 1]);
	}

	// change according to type
	public int querry(int i, int a, int b)
	{
		if(b < lo[i] || a > hi[i]) return Integer.MAX_VALUE;
		if(a <= lo[i] && b >= hi[i]) return data[i] + delta[i];
		prop(i);
		int left = querry(2 * i, a, b);
		int right = querry(2 * i + 1, a, b);
		track(i);
		return Math.min(left, right);
	}

	public void update(int i, int a, int b, int val)
	{
		if(b < lo[i] || a > hi[i]) return;
		if(a <= lo[i] && b >= hi[i])
		{
			delta[i] += val;
			return;
		}
		prop(i);
		update(2 * i, a, b, val);
		update(2 * i + 1, a, b, val);
		track(i);
	}
}

public static int gcd(int a, int b)
{
	return b == 0 ? a : gcd(b, a % b);
}

// Returns b inverse mod a in res[1].
// Note - value returned could be negative.
public static long[] extendedEuclideanAlgorithm(long a, long b)
{
	if(b==0) return new long[]{1, 0, a};
	long q = a / b;
	long r = a % b;
	long[] rec = extendedEuclideanAlgorithm(b,r);
	return new long[]{rec[1], rec[0] - q * rec[1], rec[2]};
}

public static boolean[] primeSieve(int n)
{
	boolean[] isPrime = new boolean[n - 1];
	Arrays.fill(isPrime, true);
	for(int i = 2; i * i <= n; i++)
		if(isPrime[i - 2])
			for(int j = 2 * i; j <= n; j+=i)
				isPrime[j - 2] = false;
	return isPrime[n];
}

// For p = 5, res = number of 0's at the end of n!
public static int numTimesDivide(int n, int p)
{
	int res = 0;
	while(n > 0)
	{
		res += n / p;
		n /= p;
	}
	return res;
}

// LCM/GCF = highest/lowest exp of common primes
public static ArrayList<Pair> primeFact(int n)
{
	ArrayList<Pair> res = new ArrayList<>();
	int div = 2;
	while(div * div <= n)
	{
		int exp = 0;
		while(n % div == 0)
		{
			n /= div;
			exp++;
		}
		if(exp > 0) res.add(new Pair(div, exp));
		div++;
	}
	if(n > 1) res.add(new Pair(n, 1));
	return res;
}

public static int numDivs(int n)
{
	ArrayList<Pair> divs = primeFact(int n);
	int num = 1;
	for(Pair p : divs)
		num *= p.y + 1;
	return num;
}

public static int sumDivs(int n)
{
	ArrayList<Pair> divs = primeFact(int n);
	int sum = 1;
	for(Pair p : divs)
		sum *= (Math.pow(p.x, p.y + 1) - 1) / (p.x - 1);
	return sum;
}

public static int[] MCSS(int[] a)
{
	int max = Integer.MIN_VALUE, sum = 0, start = 0, end = 0, tmp = 0;
	for(int i = 0; i < a.length; i++)
	{
		sum += a[i];
		if(sum > max)
		{
			max = sum;
			start = tmp;
			end = i;
		}
		if(sum < 0)
		{
			tmp = i+1;
			sum = 0;
		}
	}
	return new int[]{max, start, end};
}

// 2D Cumulative Frequency
static int[][] sum;
public static void rangeSum(int[][] a)
{
	sum = new int[a.length + 1][a[0].length + 1];
	for(int i = 0; i < a.length; i++)
	{
		for(int j = 0; j < a[i].length; j++)
			sum[i + 1][j + 1] = a[i][j] + sum[i + 1][j];
		for(int j = 0; j < a[i].length; j++)
			sum[i + 1][j + 1] += sum[i][j + 1];
	}
}
public static int query(int a, int b, int c, int d) // a <= b, c <= d, a <= c, b <= d
{
	return sum[c][d] - sum[a][d] - sum[c][b] + sum[a][b];
}

class Point implements Comparable<Point>
{
	static double refX, refY;
	double x, y;

	public Point(double x, double y)
	{
		this.x = x;
		this.y = y;
	}

	public Point delta(Point o)
	{
		return new Point(o.x - this.x, o.y - this.y);
	}

	// Convex Hull turn: -1 = left, 0 = linear, 1 = right
	public int turn(Point mid, Point next)
	{
		Vector u = new Vector(this, mid);
		Vector v = new Vector(mid, next);
		return new Double(u.cross(v)).compareTo(0.0);
	}

	// Convex Hull sort
	public int compareTo(Point o)
	{
		Point ref = new Point(refX, refY);
		Vector u = new Vector(ref, this);
		Vector v = new Vector(ref, o);

		if(u.isZero()) return -1;
		if(v.isZero()) return 1;
		Double product = new Double(u.cross(v));
		return product.compareTo(0.0) != 0 ?
			// change > to < for left turns
			product.compareTo(0.0) > 0 ? -1 : 1 :
			u.m < v.m ? -1 : 1;
	}

	// public int compareTo(Point o)
	// {
		// int compX = new Double(this.x).compareTo(o.x);
		// int compY = new Double(this.y).compareTo(o.y);
		// return compX == 0 ? compY : compX;
	// }

	public boolean equals(Object o)
	{
		if(o instanceof Point)
		{
			Point p = (Point)o;
			return new Double(this.x).equals(p.x) && new Double(this.y).equals(p.y);
		}
		return false;
	}
}

class Vector
{
	double x, y, m, normx, normy, theta;

	public Vector(double x, double y)
	{
		this.x = x;
		this.y = y;
		this.m = Math.sqrt(x * x + y * y);
		this.normx = x / m;
		this.normy = y / m;
		this.theta = Math.acos(normx / m);
	}

	public Vector(Point p)
	{
		this(p.x, p.y);
	}

	public Vector(Point p, Point q)
	{
		this(p.delta(q));
	}

	public boolean isZero()
	{
		return x == 0 && y == 0;
	}

	public void scale(double s)
	{
		x *= s; y *= s; m *= s;
	}

	public Vector ortho(Vector v)
	{
		return new Vector(-v.y, v.x);
	}

	public double dot(Vector o)
	{
		return (this.x * o.x + this.y * o.y) / (this.m * o.m);
	}

	// magnitude = area of parallelogram, three points are collinear if this is 0
	public double cross(Vector o)
	{
		return this.x * o.y - this.y * o.x;
	}

	public double angle(Vector o)
	{
		return Math.acos(this.dot(o));
	}

	public Vector rotate(double phi)
	{
		return new Vector(m * Math.cos(theta + phi), m * Math.sin(theta + phi));
	}
}

class Line
{
	Point p;
	Vector v;

	public Line(Point p, Vector v)
	{
		this.p = p; this.v = v;
	}

	public Line(Point p, Point q)
	{
		this(p, new Vector(p, q));
	}

	public Point next(double t)
	{
		return new Point(p.x + v.x * t, p.y + v.y * t);
	}

	// Dist from p to line
	public double dist(Point p)
	{
		Vector t = new Vector(this.p, p);
		Vector rv = v.rotate(-v.theta);
		Vector rt = t.rotate(-v.theta);
		// if segment
		if(rt.x < rv.x) return t.m;
		else if(rt.x > rv.m) return new Vector(new Point(v.x, v.y), p).m;
		// else not segment
		return rt.y - rv.y;
	}

	// returns t's for both lines
	public Point intersect(Line l)
	{
		Vector c = new Vector(this.p.x - l.p.x, this.p.y - l.p.y);
		double e = 1e-9;
		double numx = l.v.cross(c);
		double numy = this.v.cross(c);
		double denom = this.v.cross(l.v);
		if(Math.abs(denom) > e) // intersection, check starts and fractions on [0, 1] for segments
			return new Point(numx / denom, numy / denom);
		else if(Math.abs(numx) > e) // either numerator = 0, parallel
			return new Point(Integer.MIN_VALUE, Integer.MIN_VALUE);
		return new Point(Integer.MAX_VALUE, Integer.MAX_VALUE); // coincidental, check segments for overlap
	}

	// returns x component of intersections
	public Point intersect(Circle C)
	{
		double e = 1e-9;
		double a = v.x + p.y;
		double b = 2 * (p.x * v.x + p.y * v.y - a);
		double c = p.x * p.x + p.y * p.y + C.p.x * C.p.x + C.p.y * C.p.y -
					 (C.r.m * C.r.m + 2 * (p.x * C.p.x + p.y * C.p.y));
		double root = b * b - 4 * a * c;
		if(root >= e)
			return new Point((-b - Math.sqrt(root)) / (2 * a), (-b + Math.sqrt(root)) / (2 * a));
		return new Point(Integer.MIN_VALUE, Integer.MIN_VALUE);
	}

	// returns points of intersection
	public Point[] intersect(Polygon p)
	{
		Set<Point> pts = new HashSet<>();
		for(int i = 0; i < p.edges; i++)
		{
			Line e = new Line(p.vertices[i], p.vertices[(i + 1) % p.edges]);
			Point inter = this.intersect(e);
			if(inter.x >= 0 && inter.x <= 1 && inter.y >= 0 && inter.y <= 1)
			{
				pts.add(this.next(inter.x));
			}
		}
		return pts.toArray(new Point[0]);
	}
}

class Circle
{
	Point p;
	Vector r;

	public Circle(Point p, Vector r)
	{
		this.p = p;
		this.r = r;
	}

	public Circle(Point p, double r)
	{
		this(p, new Vector(r, 0.0));
	}

	public double dist(Circle c)
	{
		return new Vector(this.p, c.p).m;
	}

	public double arc(double theta)
	{
		return theta * r.m;
	}

	public double sector(double theta)
	{
		return theta / 2 * r.m * r.m;
	}

	public double segment(Point a, Point b)
	{
		double theta = new Vector(p, a).angle(new Vector(p, b));
		return (theta - Math.sin(theta)) / 2 * r.m * r.m;
	}

	public Point[] intersect(Circle c)
	{
		Point[] pts = new Point[2];
		double dist = this.dist(c);
		if(dist > this.r.m + c.r.m)
		{
			pts[0] = new Point(Integer.MAX_VALUE, Integer.MAX_VALUE);
			pts[1] = new Point(Integer.MAX_VALUE, Integer.MAX_VALUE);
		}
		else if(dist < Math.abs(this.r.m - c.r.m))
		{
			pts[0] = new Point(Integer.MIN_VALUE, Integer.MIN_VALUE);
			pts[1] = new Point(Integer.MIN_VALUE, Integer.MIN_VALUE);
		}
		else
		{
			double a = (this.r.m * this.r.m - c.r.m * c.r.m) / (2 * this.dist(c));
			double theta = Math.acos(a / r.m);
			Vector up = r.rotate(theta);
			Vector down = r.rotate(-theta);
			pts[0] = new Point(p.x + up.x, p.y + up.y);
			pts[1] = new Point(p.x + down.x, p.y + down.y);
		}
		return pts;
	}

	public Point[] intersect(Polygon p)
	{
		Set<Point> pts = new HashSet<>();
		for(int i = 0; i < p.edges; i++)
		{
			Line e = new Line(p.vertices[i], p.vertices[(i + 1) % p.edges]);
			Point inter = e.intersect(this);
			if(inter.x >= e.p.x && inter.x <= e.v.x)
				pts.add(e.next((inter.x - e.p.x) / e.v.x));
			if(inter.y >= e.p.x && inter.y <= e.v.x)
				pts.add(e.next((inter.y - e.p.x) / e.v.x));
		}
		return pts.toArray(new Point[0]);
	}
}

class Polygon
{
	int edges;
	Point[] vertices;

	public Polygon(int n, int[] vertices)
	{
		edges = n;
		for(int i = 0; i < n; i++)
			this.vertices[i] = new Point(vertices[i * 2], vertices[i * 2 + 1]);
	}

	public double area()
	{
		double a = 0;
		for(int i = 0; i < edges; i++)
			a += new Vector(vertices[i]).cross(new Vector(vertices[(i + 1) % edges]));
		return a / 2;
	}

	public boolean contains(Point p)
	{
		double e = 1e-9;
		double a = 0;
		for(int i = 0; i < edges; i++)
			if(vertices[i].equals(p))
				return true;
		for(int i = 0; i < edges; i++)
		{
			Vector u = new Vector(p, vertices[i]);
			Vector v = new Vector(p, vertices[(i + 1) % edges]);
			a += u.angle(v);
		}
		return a < e ? false : true;
	}

	// Separating axis theorem
	// returns vector to separate collision
	public Vector intersect(Polygon p)
	{
		Vector overlap = new Vector(1, 0);
		overlap.scale(Integer.MAX_VALUE);
		for(int i = 0; i < this.edges; i++)
		{
			Vector axis = new Vector(this.vertices[i], this.vertices[(i + 1) % this.edges]);
			axis = axis.ortho(axis);
			double[] projs = satHelper(axis, p);
			if(projs[0] > projs[3] || projs[1] < projs[2])
				return new Vector(0, 0);
			double oamt = Math.min(Math.abs(projs[0] - projs[3]), Math.abs(projs[1] - projs[2]));
			if(oamt < overlap.m)
			{
				overlap = new Vector(axis.normx, axis.normy);
				overlap.scale(oamt);
			}
		}
		for(int i = 0; i < p.edges; i++)
		{
			Vector axis = new Vector(p.vertices[i], p.vertices[(i + 1) % p.edges]);
			axis = axis.ortho(axis);
			double[] projs = satHelper(axis, p);
			if(projs[0] > projs[3] || projs[1] < projs[2])
				return new Vector(0, 0);
			double oamt = Math.min(Math.abs(projs[0] - projs[3]), Math.abs(projs[1] - projs[2]));
			if(oamt < overlap.m)
			{
				overlap = new Vector(axis.normx, axis.normy);
				overlap.scale(oamt);
			}
		}
		return overlap;
	}

	public double[] satHelper(Vector axis, Polygon p)
	{
		double[] overlap = {Double.MAX_VALUE, Double.MAX_VALUE, Double.MIN_VALUE, Double.MIN_VALUE};
		for(int j = 0; j < this.edges; j++)
		{
			Vector v = new Vector(this.vertices[j]);
			// only needed axis to be normalized
			double proj = axis.dot(v) * v.m;
			if(overlap[0] > proj) overlap[0] = proj;
			if(overlap[2] < proj) overlap[2] = proj;
		}
		for(int j = 0; j < p.edges; j++)
		{
			Vector v = new Vector(p.vertices[j]);
			// only needed axis to be normalized
			double proj = axis.dot(v) * v.m;
			if(overlap[1] > proj) overlap[1] = proj;
			if(overlap[3] < proj) overlap[3] = proj;
		}
		return overlap;
	}
}

// requires Point class defined above
public static void startIndex(int n, Point[] pts)
{
	int res = 0;
	for(int i = 1; i < n; i++)
		if(pts[i].y < pts[res].y || (pts[i].y == pts[res].y && pts[i].x < pts[res].x))
			res = i;
	Point.refX = pts[res].x;
	Point.refY = pts[res].y;
}

public static Stack<Point> grahamScan(int n, Point[] pts)
{
	Arrays.sort(pts);
	Stack<Point> hull = new Stack<>();
	hull.push(pts[0]);
	hull.push(pts[1]);

	for(int i = 2; i < n; i++)
	{
		Point cur = pts[i];
		Point mid = hull.pop();
		Point prev = hull.pop();
		// change comparison to handle left/linear/right turns
		// currently accepts right turns into hull
		while(prev.turn(mid, cur) <= 0)
		{
			if(prev.turn(mid, cur) == 0)
				break;
			mid = prev;
			prev = hull.pop();
		}
		// used if linear is not allowed in hull and points start linear
		if(prev.turn(mid, cur) == 0)
		{
			hull.push(prev);
			hull.push(cur);
			continue;
		}
		hull.push(prev);
		hull.push(mid);
		hull.push(cur);
	}
	return hull;
}

public static int fib(int n)
{
	int a = 0, b = 1;
	for(int i = 2; i < n + 1; i++)
	{
		b = a + b;
		a = b - a;
	}
	return b;
}

public static int binom(int n, int k)
{
	int[][] pascal = new int[2][n + 1];
	Arrays.fill(pascal[0], 1);
	Arrays.fill(pascal[1], 1);
	for(int i = 2; i < n + 1; i++)
		for(int j = 1; j < i; j++)
			pascal[i % 2][j] = pascal[(i + 1) % 2][j - 1] + pascal[(i + 1) % 2][j];
	return pascal[n % 2][k];
}

// If you impose an ordering to the sequences
// then this returns the last sequence
// Reverse/negate array or run backwards for lds
public static int lis(int[] vals)
{
	int[] dp = new int[vals.length];
	int[] recon = new int[vals.length];
	int len = 0;
	for(int i = 0; i < vals.length; i++)
	{
		int j = Arrays.binarySearch(dp, 0, len, vals[i]);
		if(j < 0) j = -(j + 1);
		if(j == len || j == len - 1) recon[j] = i;
		if(j == len) len++;
		dp[j] = vals[i];
	}
	return len; // return recon for lis indices
}

public static int lcs(String x, String y)
{
	int i = 0, j = 0;
	int[][] table = new int[x.length() + 1][y.length() + 1];
	for(i = 0; i < x.length(); i++)
		table[i][0] = 0;
	for(i = 0; i < x.length(); i++)
		table[0][i] = 0;
	for(i = 1; i <= x.length(); i++)
		for(j = 1; j <= y.length(); j++)
			if(x.charAt(i - 1) == y.charAt(i - 1)) table[i][j] = 1 + table[i - 1][j - 1];
			else table[i][j] = Math.max(table[i][j - 1], table[i - 1][j]);
	// Reconstruct sequence
	// char[] recon = new char[table[i][j]];
	// while(table[i][j] != 0)
	// {
			// if(table[i][j] != table[i - 1][j] && table[i][j] != table[i][j - 1])
				// recon[table[--i][--j]] = x.charAt(i);
			// else if(table[i][j] == table[i - 1][j]) i--;
			// else if(table[i][j] == table[i][j - 1]) j--;
	// }
	// return new String(recon);
	return table[x.length()][y.length()];
}

public static int coinChange(int[] denoms, int v)
{
	int[] ways = new int[v + 1];
	ways[0] = 1;
	for(int i = 0; i < denoms.length; i++)
		for(int j = denoms[i]; j <= v; j++)
			ways[j] += ways[j - denoms[i]];
	return ways[v];
}
public static int minChange(int[] denoms, int v)
{
	int oo = Integer.MAX_VALUE;
	int[] minWay = new int[v + 1];
	minWay[0] = 0;
	for(int i = 1; i <= v; i++) minWay[i] = oo;
	for(int i = 1; i <= v; i++)
	{
		for(int j = 0; denoms[j] <= i; j++)
		{
			int sub = minWay[i - denoms[j]];
			if(sub != oo && sub + 1 < minWay[i])
				minWay[i] = sub + 1;
		}
	}
	return minWay[v];
}

// For best size knapsack, remove -1s and dp[j] >= 0, return dp[max value]
public static int zo(int[] ws, int[] vs, int k)
{
	int[] dp = new int[k + 1];
	for(int i = 1; i < k + 1; i++) dp[i] = -1;
	for(int i = 0; i < ws.length; i++)
		for(int j = k - ws[i]; j >= 0; j--) // reverse for 0/oo knapsack
			if(dp[j] >= 0 && dp[j] + vs[i] > dp[j + ws[i]])
				dp[j + ws[i]] = dp[j] + vs[i];
	return dp[k];
}
public static int[] subsetSum(int[] s, int t)
{
	int[] subset = new int[t + 1];
	for(int i = 0; i < s.length; i++)
		for(int j = t; j >= s[i]; j--) // reverse for replacement subset
			if(subset[j - s[i]] != 0 || j == s[i])
				subset[j] = s[i];
	if(subset[t] != 0) return subset;
	return null;
}

static int[][] memo;
public static int minMult(int[][] dim, int s, int e)
{
	// If we've solved it, return the answer!
	if (memo[s][e] != -1) return memo[s][e];
	// No work to be done since the matrix itself is the answer.
	if (s == e) return 0;

	int best = minMult(dim, s + 1, e) + dim[s][0] * dim[s][1] * dim[e][1];
	for (int split = s + 1; split < e; split++)
	{
		int left = minMult(dim, s, split);
		int right = minMult(dim, split + 1, e);
		int cost = left + right + dim[s][0] * dim[split][1] * dim[e][1];
		best = Math.min(best, cost);
	}

	return memo[s][e] = best;
}

static int count;
static int[] items;
static boolean[] used;
void combo(int in, int choose)
{
	if(choose == 0) print(used);
	else if(in < count)
	{
		used[in] = true;
		combo(in + 1, choose - 1);
		used[in] = false;
		combo(in + 1, choose);
	}
}

static int count;
static int[] items, perm;
static boolean[] used;
void permute(int in)
{
	if(in > count) print(perm);
	else
	{
		for(int i = 0; i < count; i++)
		{
			if(!used[i])
			{
				used[i] = true;
				perm[in] = items[i];
				permute(in + 1);
				used[i] = false;
			}
		}
	}
}

int low = some safe low value;
int high = some safe high value;
int mid = (high - low) / 2 + low;
int iters = 50;
for(int i = 0; i < iters; i++)
{
	if(mid > target) high = mid;
	if(mid < target) low = mid;
	mid = (high - low) / 2 + low;
}
return mid;

class BinaryLift
{
	int D;
	int[] depth;
	int[][] parents;

	public BinaryLift(ArrayList<ArrayList<Integer>> graph)
	{
		D = Integer.numberOfTrailingZeros(Integer.highestOneBit(graph.size()));
		depth = new int[graph.size()];
		parents = new int[D + 1][graph.size()];
		for(int[] p : parents) Arrays.fill(p, -1);

		ArrayDeque<Integer> q = new ArrayDeque<>();
		boolean[] seen = new boolean[graph.size()];
		q.offer(0);
		seen[0] = true;
		while(!q.isEmpty())
		{
			int i = q.poll();
			for(int j : graph.get(i))
			{
				if(!seen[j])
				{
					seen[j] = true;
					parents[0][j] = i;
					depth[j] = depth[i] + 1;
					q.offer(j);
				}
			}
		}

		for(int d = 1; d <= D; d++)
			for(int i = 0; i < graph.size(); i++)
				if(parents[d - 1][i] != -1)
					parents[d][i] = parents[d - 1][parents[d - 1][i]];
	}

	public int walk(int i, int k)
	{
		for(int d = 0; d <= D && i != -1; d++)
			if(((1 << d) & k) > 0)
				i = parents[d][i];
		return i;
	}

	public int lca(int i, int j)
	{
		if(depth[i] > depth[j]) i = walk(i, depth[i] - depth[j]);
		if(depth[j] > depth[i]) j = walk(i, depth[j] - depth[i]);
		if(i == j) return i;
		for(int d = D; d >= 0; d--)
		{
			if(parents[d][i] != parents[d][j])
			{
				i = parents[d][i];
				j = parents[d][j];
			}
		}
		return parents[0][i];
	}
}
