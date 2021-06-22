package it.unibo.neat

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import org.encog.ml.data.basic.BasicMLData
import org.encog.neural.neat.NEATNetwork

class ScafiNeatHopCountUsage
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with FieldUtils {
  private val howMany = 16.0 //helps to normalize nodes..
  override def main(): Any = {
    val regression = node.get[NEATNetwork]("regression")
    rep(howMany) { data => {
      val elements = excludingSelf.reifyField(nbr(data)).values.toSeq :+ howMany
      val min = elements.min
      val normalized = min / howMany
      val inputData = new BasicMLData(Array(normalized))
      val output = if(node.has("source")) { 0.0 } else { regression.compute(inputData).getData(0) }
      node.put("output", Math.round(output * howMany))
      output
    }}
  }
}
