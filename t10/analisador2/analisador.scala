import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf


object Analisador {

  // Args = path/to/text0.txt path/to/text1.txt
  def main(args: Array[String]) {

    // create Spark context with Spark configuration
    val sc = new SparkContext(new SparkConf().setAppName("Contagem de Palavra"))

    println("TEXT1")

    // read first text file and split into lines
    val lines1 = sc.textFile(args(0))

    // TODO: contar palavras do texto 1 e imprimir as 5 palavras de mais de 3 letras com as maiores ocorrencias (ordem DECRESCENTE)

    // Considerar letras maiúsculas e minúsculas considerar como a mesma letra, dividir o texto a partir dos espaços e limpar a pontuação
    val words1 = lines1.flatMap(line => line.toLowerCase.replaceAll("[,.!?:;]","").split(" "))

    // Considerar somente as palavras com mais de 3 letras
    val more3_words1 = words1.filter(word => {word.length() > 3})

    // Inicializar palavras e contar palavras do texto 1
    val count_words1 = more3_words1.map(word => (word, 1)).reduceByKey{case (word1,word2) => word1 + word2}

    // Ordenar DECRESCENTEMENTE as ocorrencias
    val desc_words1 = count_words1.sortBy(word => (-word._2,word._1))
    // val desc_words1 = count_words1.map(word => word.swap).sortByKey(false).map(word => word.swap)

    // 5 palavras com as maiores ocorrencias
    val top5_words1 = desc_words1.take(5)

    // imprimir na cada linha: "palavra=numero"
    top5_words1.foreach(word => println(word._1 + "=" + word._2))

    println("TEXT2")

    // read second text file and split each document into words
    val lines2 = sc.textFile(args(1))

    // TODO: contar palavras do texto 2 e imprimir as 5 palavras com as maiores ocorrencias (ordem DECRESCENTE)
    // Considerar letras maiúsculas e minúsculas, dividir o texto a partir dos espaços e limpar a pontuação
    val words2 = lines2.flatMap(_.toLowerCase.replaceAll("[,.!?:;]","").split(" "))

    // Considerar somente as palavras com mais de 3 letras
    val more3_words2 = words2.filter({_.length() > 3})

    // Inicializar palavras e contar palavras do texto 2
    val count_words2 = more3_words2.map((_, 1)).reduceByKey{_ + _}

    // Ordenar DECRESCENTEMENTE as ocorrencias
    val desc_words2 = count_words2.sortBy(word => (-word._2,word._1))

    // 5 palavras com as maiores ocorrencias
    val top5_words2 = desc_words2.take(5)

    // imprimir na cada linha: "palavra=numero"
    top5_words2.foreach(word => println(word._1 + "=" + word._2))

    println("COMMON")

    // TODO: comparar resultado e imprimir na ordem ALFABETICA todas as palavras que aparecem MAIS que 100 vezes nos 2 textos
    val more100_words1 = desc_words1.filter(word => {word._2 > 100}).map(word => word._1)

    val more100_words2 = desc_words2.filter(word => {word._2 > 100}).map(word => word._1)

    val words1_words2 = more100_words1.intersection(more100_words2).collect().toList.sorted

    // imprimir na cada linha: "palavra"
    words1_words2.foreach(println)
  }

}
