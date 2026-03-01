'use client'

import { motion } from 'framer-motion'
import TextHeading from '@/components/ui/text-heading/text-heading'
import Text from '@/components/ui/text/text'
import { StackVertical } from '@/components/layout/layout-stack/layout-stack'
import Image from 'next/image'
import Ruler from '@/components/ui/ruler/ruler'

export function HeroSection() {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="relative pb-4"
        >
            <div className="flex flex-col sm:flex-row sm:items-center gap-5 mb-6">
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.4 }}
                    className="flex-shrink-0"
                >
                    <Image
                        src="/image.png"
                        alt="Rohit Krishnan Somasundaram"
                        width={140}
                        height={140}
                        className="rounded-full object-cover ring-2 ring-purple-500/30"
                        priority
                    />
                </motion.div>

                <StackVertical gap="xs">
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.2 }}
                    >
                        <TextHeading as="h1" className="font-bold text-2xl sm:text-3xl md:text-4xl lg:text-5xl">
                            Hi! I am Rohit.
                        </TextHeading>
                    </motion.div>
                </StackVertical>
            </div>

            <Ruler color='colorless' marginTop='none' marginBottom='none' />

            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
            >
                <StackVertical gap="sm">
                    <Text>
                        I&apos;m a ML Engineer based in Chennai, India; currently doing MLOps at ACV Auctions. I care deeply about making ML systems that actually work in the real world: systems that are reliable, observable, and useful well beyond a notebook.
                    </Text>
                    <Text>
                        I&apos;m also someone who gets distracted by cosmology and questions about the origins of life. There&apos;s something I find endlessly interesting about the same mathematics showing up in neural networks and in the large-scale structure of the universe. This site is where I write about things I&apos;m trying to understand.
                    </Text>
                </StackVertical>
            </motion.div>
        </motion.div>
    )
}
