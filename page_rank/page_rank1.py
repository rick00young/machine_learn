from pygraph.classes.digraph import digraph

class PageRankIterator:
	__doc__ = '''计算一张图的PR值'''
	def __init__(self,dg=None):
		# 阻尼系数即α
		self.damping_factor = .85
		# 最大迭代次数
		self.max_iterations = 100
		# 确定迭代是否结束的参数即ϵ
		self.min_delta = .00001
		self.graph = dg

	def page_rank(self):
		# 先将图中没有出链的节点改为对所有节点都有出链
		for node in self.graph.nodes():
			if len(self.graph.neighbors(node)) == 0:
				for node2 in self.graph.nodes():
					digraph.add_edge(self.graph, (node, node2))

		nodes = self.graph.nodes()
		graph_size = len(nodes)

		if graph_size == 0:
			return {}

		# 给每个节点赋予初始的PR值
		page_rank = dict.fromkeys(nodes, 1.0/graph_size)
		# 公式中的(1−α)/N部分
		damping_value = (1.0 - self.damping_factor)/graph_size
		flag = False
		for i in range(self.max_iterations):
			change = 0
			for node in nodes:
				rank = 0
				# 遍历所有'入射的页面
				for incident_page in self.graph.incidents(node):
					rank += self.damping_factor * (page_rank[incident_page] / len(self.graph.neighbors(incident_page)))
				rank += damping_value
				change += abs(page_rank[node] - rank)
				page_rank[node] = rank
			print("This si NO.%s iteration" % (i+1))
			print(page_rank)
			if change < self.min_delta:
				flag = True
				break
		if flag:
			print("finished in %s iteration!" % node)
		else:
			print("finished out of 100 iteration")
		return page_rank

if __name__ == '__main__':
	dg = digraph()

	dg.add_nodes(["A", "B", "C", "D", "E"])

	dg.add_edge(("A", "B"))
	dg.add_edge(("A", "C"))
	dg.add_edge(("A", "D"))
	dg.add_edge(("B", "D"))
	dg.add_edge(("C", "E"))
	dg.add_edge(("D", "E"))
	dg.add_edge(("B", "E"))
	dg.add_edge(("E", "A"))

	pr = PageRankIterator(dg=dg)
	page_ranks = pr.page_rank()

	print("The final page rank is\n", page_ranks)