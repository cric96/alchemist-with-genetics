package it.unibo.neat

import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.interfaces.Node

interface PimpNode {
    fun <T> Node<*>.data(name : String) : T = this.getConcentration(SimpleMolecule(name)) as T
    fun Node<*>.dataDouble(name : String) : Double = this.data(name)
    fun <T> Node<T>.put(name : String, value : T) = this.setConcentration(SimpleMolecule(name), value)
}