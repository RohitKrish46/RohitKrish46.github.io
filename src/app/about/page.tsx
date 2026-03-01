'use client'

import BaseContainer from "@/components/layout/container/base-container"
import { StackVertical } from "@/components/layout/layout-stack/layout-stack"
import TextHeading from "@/components/ui/text-heading/text-heading"
import { SectionFooter } from "@/components/layout/footer/SectionFooter"
import Text from "@/components/ui/text/text"
import { DynamicBreadcrumb } from "@/components/ui/primitives/breadcrumb"
import { ThemeToggle } from "@/components/ui/theme/theme-toggle"

export default function About() {
    return (
        <BaseContainer size="md" paddingX="md" paddingY="lg">
            <StackVertical gap="md">
                <div className="flex items-center justify-between">
                    <DynamicBreadcrumb
                        items={[
                            { href: '/', label: 'Home', emoji: '👾' },
                            { label: 'About' }
                        ]}
                    />
                    <ThemeToggle />
                </div>

                <div>
                    <TextHeading as="h1" weight="bold">
                        About Me
                    </TextHeading>
                    <Text variant="muted" size="xs" className="mb-8">Rohit Krishnan Somasundaram</Text>
                    <StackVertical gap="md">
                        <Text>
                            Hi! Thanks for stopping by. I&apos;m a ML Engineer based in Chennai, India, working at the intersection of machine learning, software engineering, and generative AI.
                        </Text>

                        <Text>
                            My path into ML wasn&apos;t a straight line. I started with a Bachelor&apos;s in Computer Science at Sairam Engineering College in Chennai, where I first got a taste of algorithms and software. It was during my Master&apos;s in Computer Science at Simon Fraser University — specializing in Visual Computing — that I found the thing that truly hooked me: the mathematics and intuition behind how machines learn to see and understand the world.
                        </Text>

                        <Text>
                            After SFU, I joined Prodapt Solutions as an Associate Software Engineer, which grounded me in production systems. Then I made the full jump into ML — first as a Research Intern at the Medical Image Analysis Lab at SFU applying deep learning to medical imaging, then into industry roles at VisionBox Inc. and AmiableAi Inc., building and shipping AI systems in computer vision and NLP.
                        </Text>

                        <Text>
                            What drives me isn&apos;t just the engineering — it&apos;s the questions underneath. How does intelligence arise? What does it mean for a system to understand something? These questions pull me toward cosmology and the origins of life as much as toward ML research. I find the boundary between physics, biology, and computation endlessly fascinating.
                        </Text>

                        <Text>
                            This site is where I think out loud — about ML, math, and ideas I&apos;m working through. If any of it resonates, I&apos;d love to hear from you.
                        </Text>

                        <Text>
                            You can reach me at{' '}
                            <a href="mailto:krishr.somu@gmail.com" className="text-purple-500 hover:underline">krishr.somu@gmail.com</a>
                            {' '}or find me on{' '}
                            <a href="https://github.com/RohitKrish46" className="text-purple-500 hover:underline" target="_blank" rel="noopener noreferrer">GitHub</a>
                            ,{' '}
                            <a href="https://linkedin.com/in/rohit-krishnan-s" className="text-purple-500 hover:underline" target="_blank" rel="noopener noreferrer">LinkedIn</a>
                            , or{' '}
                            <a href="https://x.com/KrishrSomu" className="text-purple-500 hover:underline" target="_blank" rel="noopener noreferrer">Twitter</a>
                            .
                        </Text>
                    </StackVertical>
                </div>
            </StackVertical>
            <SectionFooter color="purple" />
        </BaseContainer>
    )
}
