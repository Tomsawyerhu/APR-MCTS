from neo4j import GraphDatabase


def METHOD_QUERY(full_qualified_name, simple_name):
    """
    找到方法节点
    :param full_qualified_name: 方法的完全限定名
    :param simple_name: 方法名
    :return:
    """
    return (
        "MATCH (n:Method)\n"
        f"WHERE n.full_qualified_name = '{full_qualified_name}'\n"
        f"and n.name = '{simple_name}'\n"
        f"RETURN n"
    )


def ClAZZ_QUERY(full_qualified_name):
    """
    找到类节点
    :param full_qualified_name: 类的完全限定名
    :return:
    """
    return (
        "MATCH (n:Clazz)\n"
        f"WHERE n.full_qualified_name = '{full_qualified_name}'\n"
        f"RETURN n"
    )


def METHOD_LIST_QUERY(full_qualified_name):
    """
    找到类中定义的方法列表
    :param full_qualified_name: 类的完全限定名
    :return:
    """
    return (
        "MATCH (a:Clazz)-[r:CONTAIN]-(n:Method)\n"
        f"WHERE a.full_qualified_name = '{full_qualified_name}'\n"
        "RETURN n"
    )


def IMPORT_QUERY(full_qualified_name):
    """
    找到类的import
    :param full_qualified_name: 类的完全限定名
    :return:
    """
    return (
        "MATCH (a:Clazz)-[r:IMPORT]-(n:Clazz)\n"
        f"WHERE a.full_qualified_name = '{full_qualified_name}'\n"
        "RETURN n"
    )


def METHOD2METHOD_INVOKE_QUERY(signature):
    """
    找到调用的函数(除了调用自己)
    :param signature: 函数节点的signature
    :return:
    """
    return (
        "MATCH (a:Method)-[r:INVOKE]-(n:Method)\n"
        f"WHERE a.signature = '{signature}'\n"
        f"and n.signature != '{signature}'\n"
        "RETURN n"
    )


def LOCAL_VARIABLE_QUERY(signature):
    """
    找到函数定义的局部变量
    :param signature: 函数节点的signature
    :return:
    """
    return (
        "MATCH (a:Method)-[r:CONTAIN]-(n:LocalVariable)\n"
        f"WHERE a.signature = '{signature}'\n"
        "RETURN n"
    )

def PARENT_CLASS_QUERY(full_qualified_name):
    """
    找到父类
    :param full_qualified_name: 子类名
    :return:
    """
    return (
        "MATCH (a:Clazz)-[r:EXTEND]-(n:Clazz)\n"
        f"WHERE a.full_qualified_name = '{full_qualified_name}'\n"
        "RETURN n"
    )

class Neo4jClient:
    def __init__(self,
                 uri="bolt://localhost:7687",
                 username="neo4j",
                 password="12345678"):
        """
        初始化 Neo4j 客户端
        :param uri: Neo4j Bolt 协议地址（如 "bolt://localhost:7687"）
        :param username: 用户名
        :param password: 密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        """
        关闭数据库连接
        """
        if self.driver:
            self.driver.close()

    def execute_query(self, query, parameters=None):
        """
        执行 Cypher 查询并返回结果
        :param query: Cypher 查询语句
        :param parameters: 查询参数（可选）
        :return: 查询结果
        """
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def print_results(self, results):
        """
        打印查询结果
        :param results: 查询结果列表
        """
        for record in results:
            print(record)
