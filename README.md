我们是中国太极计划重庆大学物理学院“重庆大学分级”小组 前一排提示：有点口水话，所以看官们嫌烦可以AI总结一下 这是由我们重庆大学物理学院四个本科学生关于2024年太极科研训练计划的训练作业的上传。由于整个从广义相对论理论计算大质量天体产生引力波，并且研究得到引力波的性质并相应的进行引力波探测装置的设计，再到数据分析处理这整个过程涉及具体的内容很多，且理论难度很深，加之时间非常有限，我们将主要精力放在了具体的数据处理上，并对于每一个环节有一个简单的研究讨论。然后我们还各自加入了一些我们各自的想法，特别地加入了一些对于量子引力效应的讨论。因此我们的工作主要包含原定要求完成的作业和我们自己做的一些相关联的工作： 邮件里面所要求的工作： 1.邮件当中所提到的六个作业，是由我们四个人合作完成的。在这些作业之上，我们贡献不分先后，在各自的繁忙的学习和生活当中主动自觉地抽出时间，进行了充分地合作和工作。这里的上传的所有文件是在在我们每个人对于各自的部分进行负责和具体的工作之后，在最后提交和汇总修改阶段都进行了详细地讨论和思考（毕竟我们深刻地理解这六个作业是用来让我们学习和了解引力波数据处理的各种方法，会遇到的问题以及各种其他的技术细节。）都各自产生了理解并且也在所有的工作下面都留下了自己的独立见解和批注。我们在尽可能地保持每个人的独特的思考的同时，也保留了非常重要的团队精神。团队精神在大型的科学合作项目，比如引力波探测和LHC这样的大型粒子对撞机的整体性工作当中都是非常重要的。另外，由于确实很多是由于学些业确实比较繁忙，所以有很多地方做的不是很好，特别是代码上并没有做的很细致。但是我们还是严格地按照邮件里面的要求完成了任务的嘿嘿。 每一个任务的具体完成情况： 第一个大任务： 第一个子任务：跑个代码而已，就是除了中间有一个地方需要一个叫“BBH_testing_1s_8192Hz_10Ksamp_25n_iSNR2_Hdet_astromass_1seed_ts_0.sav”的文件，我们知道这就是一个引力波数据的sample，但是确实在b站和网上的那几个地址找了很久也没能找到这个原始的文件，所以我们就从LIGO官网上面下载了GW150914的原始文件的sample来作为训练的数据分析对象。最后的结果上传在了repo当中。（后面也想到了其实就是应该在网上跑通这个代码，所以我们也上传了网上跑通之后的代码） 第二个子任务：这个就是将第一个子任务里面所使用的神经网络模型换成了Restnet(用的152的型号)，并且思考了如何从得到双黑洞的引力波的较为强的引力波信号，转向了去获取精度要求更高的，双中子星的较弱的引力波信号的方法。我们认为虽然肯定有很大改进空间，但是确实至少我们按照要求完成了这一任务。（这一次我们上传的是直接在网上进行运行之后得到的代码了，而非本地的） 第二个大任务：这个任务确实也是比较地简单，正如选拔作业说明文档里面的描述那样，我们只需要修改这里面的一个GPS时间就可以完成任务，不过我们还是集体地讨论并且略微整理改进了一下这里面的代码。最后也是成功地跑出来了代码，并且上传了。（一样是使用了网上的原始代码跑通了然后上传了的。） 第三个大任务：这个部分其实要详细做好和学清楚确实工作量很大，而且主要负责翻译的同学在做这个东西的翻译的时候就一直在考虑如何证明自己是人工翻译出来的问题。我相信很多的其他组的同学肯定都是会较为大量地借用各种翻译工具来辅助完成这项任务（尽管有提出不要使用的要求），实际上，我们在最后的校准和对于这些文档的里面引用的一些其他的技术性文档里面的内容的简单理解的时候确实就是直接使用了AI工具进行总结。但是具体的翻译还是一位同学在不使用任何大规模的翻译软件的情况下手动翻译的，确实做了很久，也写了比较长的一段最后面的总结。（然后校准是由另一名同学完成的，）在这整个最后上传的7个ipynb文档汇总的一个大的ipynb文件的开头，有一些专业化翻译该有的说明，请审批作业的老师可以稍微注意看两眼。 第四个大任务：这个是在六个任务难度里面标注最高的一个，而且在我们一开始拿到代码的时候，就意识到了这个应该是一个那种专业研究计算算法的大型研究，所以其中这样一个大项目当中的很多代码都是互相引用和有关联的。因此在最开始的时候我们询问了王赫老师是否需要对所有的进行一个解读和翻译。不过也确实是由于时间和各方面的问题，我们最后只是进行了题目当中所要求的两个matlab算法文件向python的转化。至于优化，我们做了一些尝试，详情请见具体文档。 第五个大任务：尽管并没有标注成为最为困啊的一项任务，但是我认为这项任务确实也简单。我们小组在一起学习了小波变换之后，通过上网查询资料一件各个方面，已经成功地想到办法使用其他的包里面所提供的小波变换的代码实现了小波变换，并且得到了图形（具体文件在文件夹目录里面一起）。然后我们就研究了题目所要求使用的程序包。这个程序包是没有什么他人的实用例子的，所以我们自己进行了里面文档的剖析，并且成功地运用里面所提到的函数，进行了要求的小波变换（另外，题目当中所给的.h5文件中含有好几个引力波波形文件，依照我们的英语知识，我们选取了OBS文件夹中的数据集作为所处理的数据集【另外的几个的英文名称在我们的讨论下都一致认为应该是已经经过处理之后的】）并且分别对里面x,y,z分别于t进行了时频可视化分析和小波变换。同时我们还是用网站下载软件将题目中所提到的LISA data challenge的网站进行了全网站下载，并且使用的AI工具分析了其中的信息。我们认为我们满足了选拔作业当中的要求。 2.文件当中我们还提交了我们自己的一些额外的工作（我们的确在这个过程中大量地使用了AI工具和各种其他方面的帮助，但是我们认为在高精尖的复杂科学研究领域去借助一切帮助是一件很合理的事情。）关于我们所做的额外工作的详细介绍，有一个专门的文件展开来讲述 主要包括： （1）李宗泽同学对于量子引力的相关问题。经过了长时间的兴趣方面的探索和思考，以及一步一步地进行了对于物理的系统性学习。下面是他的三个想法，其中两个在我们这较短的时间内有所较多的工作内容，另外一个基本上只能处于设想阶段（主要还是因为这项工作所要求的工作量有点太大了，但是确实是一个可以实现的想法）： 1.尽管弦理论的数学非常优美，也确实一直有很多物理学家，甚至是一些数学家（比如丘成桐）在这方面做了很多即使没有物理意义，但也是绝对有很重要的数学意义的工作，而且能够在基本原理上某种程度上那些在科幻小说当中所描述的各种奇思妙想的可能性。（至少弦理论是一个数学上非常有趣的理论）但是其预言要么就是没有给出很准确的一些关于一些具体的在一些比较容易的领域能够被观测和讨论的内容，要么就是所需要的条件过于苛刻（比如基本上就不太可能接近的黑洞和对撞机中超高能量的情形来检测超对称粒子【而且超对称也不是量子引力专属的概念，普通的只考虑粒子物理和其他三个基本相互作用的超对称模型也是很不错的理论】，亦或者是对于零质量的引力子的检测【或许这个希望还大一些，因为过去中微子的探测也是采取的类似的方式，可以参考日本的超级神冈探测器】）。 在上大学之后拥有了足够的自由时间的情况之下，李宗泽同学经过了长时间独立的考察和整体性思考，从弦论（甚至仅仅对于超对称）的坚定的信仰和爱好者，转变成了一个理性的研究者，和思考者。实际上圈量子引力的思想更加符合他的对于如果时空以及引力是量子化的原始直观想象。（但是也不排除弦和圈这两个拓扑研究对象的本质统一性）。 然后就从对于引力量子化的原始想象和思考出发，并在考虑了现有的各项有可能探测到量子化引力的效应的实验现象的方面，认为直接地去利用和引力波探测装置同一个原理的方法来直接地探测和研究时空在普朗克尺度之下的扰动的奇异性质是最为具有可信度和能够直击量子化引力的本质的实验方向（可以首先检验是否存在量子化的引力，以及其具体的量子化的方式）。但是由于这个方向所需的精度很高（普朗克尺度大概是在10的负35次方这样一个数量级，而尽管LIGO装置的Strain并不是长度概念，但是可以想象，以及根据一篇使用弦理论目前最后比较能够得到认可的理论框架方案——Ads/Cft框架进行计算的论文，这个精度是目前短期之内难以达到和想象的），所以我们就引力波探测装置的精度和各方面做了一个简单的分析和讨论（主要是时间上不足够，不然我们能够对于更多文献资料和方案做具体的分析和想法）。然后还探讨了一些相对比较了解一些相关技术（大规模的高端航天技术，以及类卡西米尔效应和LIGO之前所发现和声称的镜子移动的那种对于量子效应的放大（主要是涉及QED，也就是电磁基本相互作用）是否能够在量子引力效应上面也能够实施。都是简单地讨论，因为相关的研究也已经比较丰富了，所以确实可能也不是什么特别了不起的事情）。 1.5：基于上面的观点，我们曾经想着应该尝试在现有的引力波探测器上面进行一下数据分析，不过在进行了一定的数据分析之后，我们就发现了其实我们很难从现有的引力波探测装置的精度水平得到我们需要分析到的数量级的数据，所以就暂时作罢了。（有一些关于各个现在已经建设好的引力波探测装置的精度分析和一些对于未来计划进行建设的引力波探测装置的设计预期的资料） 2.既然我们这个训练计划是针对于引力波数据处理的训练和研究计划，那么我们更多地去把精力和思考去放在数据处理之上是非常自然的。实际上，在参加本训练选拔之前，我们就已经对于这种大型科学装置的整体就已经有所了解，并且也明白各种机器智能相关（比如最为典型的机器学习，和目前非常火热的AI大模型中的复杂语言处理被复杂任务分析能力）的研究和方法正以一个极其重要的地位在实验领域发光发热。实际上，在2022年Chatgpt的能力惊艳了我们之后，我们就开始思考一种能够在一个相对较为专业化的领域，结合自然语言处理的强大能力和分析能力，与一个基于知识图谱和算法与物理知识库的专业多维度和多方面的分析系统的快速解析用专家系统进行结合，使得其具有强大的多方面并行数据处理和综合分析能力。这样的AI可以专门用于复杂数据系统的数据分析。这个东西只需要组织较多的专业人员对于系统所储备的物理知识公式和各中数据处理算法处进行把关，然后通过对一些比较强大的AI自然语言处理大模型进行微调（比如可以使用百度，阿里，腾讯等国内大公司的相对来说更加基于中文语言环境的大模型进行微调，让他在专业领域上面能够更好地理解专业需求）以一个API接口的方式接入我们的分析软件当中。很明显，这将极大地减少相关工作人员的工作时间，从而解放出更多的生产力到其他更加重要的工作上面。关于这个问题，我们也有做一些调查，不过由于时间有限，加上确实可能相关的资料和代码未能开源或者在一些网站内藏的比较深，所以并未能有太多的成果。 一些大家一起得到的一些想法： 正如最开始所说的，整个从广义相对论到引力波探测和数据处理是一个很复杂的过程（爱因斯坦提出这个概念之后100年才真正地探测到了引力波嘛，也是一个体现），所以想要涉及整个过程的所有部分不是说不可能，至少是我们这样的短暂的一段时间内无法完成的事情。但是我们还是想办法在网上搜索和结合老师的帮助，去寻找了一些相关方面的内容，包括： 1.一个基于SXS模拟计划数据的，引力波的在空间中的函数结果，以及另外的一个简单的双黑洞的用于模拟产生的引力波的程序，做了较为低精度的计算和于SXS给出的函数进行比较。同时还寻找并且研究了一下一个引力波从不同的空间角度经过LIGO这样的双引力波探测系统的时候会在两个引力波探测器上面产生怎么样的波形，和天基的三角形引力波探测器的波形（主要时通过寻找论文，然后我们稍微解读了一下）。 2.整理了一些文献和有用的网站地址，以作为以后研究和工作方便进行的基石。 其余的三位同学也有很多自己的看法，并且在李宗泽同学的思考当中提出了很多宝贵的意见和协助进行了具体工作当中的很大一部分。（比如算法上的帮助，资料收集和思想的进一步深化等等）我们是一个整体！ 有些关于我们收集的额外的资料因为时间问题整理起来过于麻烦，所以会在后续有时间的时候陆陆续续上传。如果各位看官感兴趣可以等待一下后续的更新。
