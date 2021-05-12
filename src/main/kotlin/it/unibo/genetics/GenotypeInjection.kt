package it.unibo.genetics

import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.interfaces.Node

interface GenotypeInjection<G> {
    fun inject (genotype : G, node : Node<Any>) : Unit
}

object GenotypeInjections {
    fun <G> moleculeInjection(name : String = "genetics") : GenotypeInjection<G> = object : GenotypeInjection<G> {
        override fun inject(genotype: G, node: Node<Any>) = node.setConcentration(SimpleMolecule(name), genotype)
    }
}