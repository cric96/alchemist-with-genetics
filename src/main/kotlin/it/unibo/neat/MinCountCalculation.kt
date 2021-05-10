package it.unibo.neat

import it.unibo.alchemist.core.implementations.Engine
import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.interfaces.Environment
import it.unibo.alchemist.model.interfaces.Node
import it.unibo.alchemist.model.interfaces.Position
import kotlin.math.abs

class MinCountCalculation(file : String, time : Double = 2.0) : AbstractAlchemistScoreCalculation(file, time), PimpNode {
    override fun evalSimulation(simulation: Engine<Any, Position<*>>): Double {
        val sourceNodes = simulation.environment.nodes.filter { it.contains(SimpleMolecule("source")) }.toSet()
        hopCountBreadthSearch(simulation.environment, sourceNodes, emptySet())
        return simulation.environment.nodes.sumOf { node -> abs(node.dataDouble("result") - node.dataDouble("output")) }
    }

    fun hopCountBreadthSearch(environment: Environment<Any, Position<*>>, toExpand : Set<Node<Any>>, visited : Set<Node<Any>>, level : Int = 0) {
        if(toExpand.isEmpty()) { return }
        val newVisited = visited.plus(toExpand)
        val newToExpand = toExpand.flatMap { environment.getNeighborhood(it) }.filterNot { visited.contains(it) }.toSet()
        toExpand.forEach { it.put("result", level) }
        hopCountBreadthSearch(environment, newToExpand, newVisited, level + 1)
    }
}