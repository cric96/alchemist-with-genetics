package it.unibo.gnn.evolutionary

import io.jenetics.{DoubleGene, Genotype}

trait NetworkCodec[E] {
  def loadFromGenotype(genotype : Genotype[DoubleGene]) : E

  def genotypeFactory(): Genotype[DoubleGene]
}
