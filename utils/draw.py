# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:38:16 2020

@author: Donal
"""
import pydot

def drawdiagram(utterancetree):

    def draw(parent_name, child_name):
        edge = pydot.Edge(parent_name, child_name)
        graph.add_edge(edge)
    
    
    
    def visit(node, parent=None):
    
        for k,v in node.items():
            if isinstance(v, dict):
                # We start with the root node whose parent is None
                # we don't want to graph the None node
                if parent:
                    draw(parent, k)
                visit(v, k)
            else:
                draw(parent, k)
                # drawing the label using a distinct name
                draw(k, k+'_'+v)
                
    graph = pydot.Dot(graph_type='graph')
    visit(utterancetree)
    graph.write_png('Diagrams\\utterance.png')