'use client'

import BaseContainer from "@/components/layout/container/base-container"
import { StackVertical } from "@/components/layout/layout-stack/layout-stack"
import TextHeading from "@/components/ui/text-heading/text-heading"
import { SectionFooter } from "@/components/layout/footer/SectionFooter"
import Text from "@/components/ui/text/text"
import { ThemeToggle } from "@/components/ui/theme/theme-toggle"
import { Navbar } from "@/components/ui/navbar/Navbar"
import Image from "next/image"
import { MapPin, Mail, Briefcase, GraduationCap, FlaskConical, Github, Linkedin, Twitter, Coffee } from "lucide-react"

const timeline = [
    {
        org: "ACV Auctions",
        role: "ML Engineer III, MLOps",
        period: "Sep 2025 – Present",
        type: "work" as const,
        current: true,
        description: "Currently building and scaling a centralized LLM Gateway, focused on making LLM access secure, reliable, and truly production-ready across the organization.",
    },
    {
        org: "Independent Research",
        role: "Career Break",
        period: "Oct 2023 – Aug 2025",
        type: "break" as const,
        description: "Took time to work on a few problems around first-principle based AI agents.",
    },
    {
        org: "AmiableAi Inc.",
        role: "AI Engineer",
        period: "Jan 2023 – Oct 2023",
        type: "work" as const,
        description: "Built and optimized end-to-end AI systems spanning LLMs, RAG pipelines, and ML infrastructure, constantly pushing toward faster, more accurate, and cost-efficient production systems.",
    },
    {
        org: "VisionBox Inc.",
        role: "AI Engineer",
        period: "Mar 2022 – Dec 2022",
        type: "work" as const,
        description: "Focused on scaling production-grade ML systems across computer vision and predictive modeling, translating research ideas into measurable real-world impact.",
    },
    {
        org: "SFU Medical Image Analysis Lab",
        role: "Research Intern",
        period: "May 2020 – Aug 2020",
        type: "research" as const,
        description: "Developed interpretable deep learning models for medical imaging, building a ResNet-based chest X-ray classification pipeline to improve diagnostic accuracy and reduce false negatives.",
    },
    {
        org: "Simon Fraser University",
        role: "M.S. Computer Science — Visual Computing",
        period: "Sep 2019 – Apr 2021",
        type: "education" as const,
        description: "Where I found what truly hooked me: the math and intuition behind how machines learn to see and understand the world.",
    },
    {
        org: "Prodapt Solutions",
        role: "Associate Software Engineer",
        period: "May 2018 – Jul 2019",
        type: "work" as const,
        description: "Where I got introduced to production-grade software engineering, grounding me in real-world systems before I went deeper into ML.",
    },
    {
        org: "Sairam Engineering College",
        role: "B.S. Computer Science",
        period: "Aug 2014 – Apr 2018",
        type: "education" as const,
        description: "Foundation in algorithms, software engineering, and systems where I first discovered my love for computation.",
    },
]

const skills = [
    "Python", "PyTorch", "TensorFlow", "Computer Vision",
    "NLP / LLMs", "Generative AI", "Deep Learning", "MLOps", "AWS", "Docker",
]

const typeConfig = {
    education: {
        Icon: GraduationCap,
        color: "text-purple-500",
        bg: "bg-purple-500/10",
        border: "border-purple-500/30",
    },
    work: {
        Icon: Briefcase,
        color: "text-green-500",
        bg: "bg-green-500/10",
        border: "border-green-500/30",
    },
    research: {
        Icon: FlaskConical,
        color: "text-amber-500",
        bg: "bg-amber-500/10",
        border: "border-amber-500/30",
    },
    break: {
        Icon: Coffee,
        color: "text-sky-500",
        bg: "bg-sky-500/10",
        border: "border-sky-500/30",
    },
}

export default function Home() {
    return (
        <BaseContainer size="lg" paddingX="md" paddingY="lg">
            <StackVertical gap="md">
                {/* Nav row */}
                <div className="flex items-center justify-between mb-8">
                    <Navbar />
                    <ThemeToggle />
                </div>

                {/* Page header */}
                <div>
                    <TextHeading as="h1" weight="bold">About Me</TextHeading>
                    <Text variant="muted" size="xs" className="mb-4">Rohit Krishnan Somasundaram</Text>
                    <Text className="mt-3">
                        ML Engineer based in Chennai, working at the intersection of machine learning,
                        computer vision, and generative AI. I care about building systems that actually work
                        and about the deeper questions of how intelligence arises in the first place.
                    </Text>
                </div>

                {/* Split layout */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-2">

                    {/* Left: Timeline */}
                    <div className="md:col-span-2 space-y-10">

                        {/* Timeline */}
                        <div>
                            <h3 className="text-sm font-semibold mb-6">Career &amp; Education</h3>
                            <div className="relative">
                                {/* Vertical line */}
                                <div className="absolute left-4 top-2 bottom-2 w-px bg-border" />

                                <div className="space-y-6">
                                    {timeline.map((item, i) => {
                                        const { Icon, color, bg, border } = typeConfig[item.type]
                                        return (
                                            <div key={i} className="relative flex gap-4">
                                                {/* Icon dot */}
                                                <div className={`relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border ${border} ${bg}`}>
                                                    <Icon className={`h-3.5 w-3.5 ${color}`} />
                                                </div>

                                                {/* Content */}
                                                <div className="pb-1 pt-0.5 flex-1 min-w-0">
                                                    <div className="flex flex-wrap items-center gap-2 mb-0.5">
                                                        <span className="text-sm font-medium">{item.org}</span>
                                                        {'current' in item && item.current && (
                                                            <span className="text-xs px-1.5 py-0.5 rounded-full bg-purple-500/15 text-purple-500 font-medium">
                                                                Current
                                                            </span>
                                                        )}
                                                    </div>
                                                    <div className="flex flex-wrap items-center gap-1.5 mb-2">
                                                        <span className={`text-xs ${color}`}>{item.role}</span>
                                                        <span className="text-xs text-muted-foreground">·</span>
                                                        <span className="text-xs text-muted-foreground">{item.period}</span>
                                                    </div>
                                                    <Text size="sm" variant="muted">{item.description}</Text>
                                                </div>
                                            </div>
                                        )
                                    })}
                                </div>
                            </div>
                        </div>

                    </div>

                    {/* Right sidebar */}
                    <div className="space-y-6 w-full min-w-0">

                        {/* Photo */}
                        <div className="relative w-full aspect-square rounded-xl overflow-hidden">
                            <Image
                                src="/Image.png"
                                alt="Rohit Krishnan Somasundaram"
                                fill
                                className="object-cover object-top"
                                priority
                            />
                        </div>

                        {/* Quick facts */}
                        <div className="w-full rounded-lg border border-border p-4 space-y-3">
                            <h3 className="text-sm font-semibold">Quick Facts</h3>
                            <div className="space-y-2">
                                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                    <MapPin className="h-3.5 w-3.5 shrink-0" />
                                    <span>Chennai, India</span>
                                </div>
                                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                    <Briefcase className="h-3.5 w-3.5 shrink-0" />
                                    <span>MLE III, MLOps @ ACV Auctions</span>
                                </div>
                                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                    <GraduationCap className="h-3.5 w-3.5 shrink-0" />
                                    <span>M.S. CS @ SFU</span>
                                </div>
                            </div>
                            <div className="pt-2 border-t border-border space-y-2">
                                <a href="mailto:krishr.somu@gmail.com" className="flex items-center gap-2 text-sm text-purple-500 hover:underline">
                                    <Mail className="h-3.5 w-3.5 shrink-0" />
                                    krishr.somu@gmail.com
                                </a>
                                <a href="https://github.com/RohitKrish46" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-sm text-purple-500 hover:underline">
                                    <Github className="h-3.5 w-3.5 shrink-0" />
                                    RohitKrish46
                                </a>
                                <a href="https://linkedin.com/in/rohit-krishnan-s" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-sm text-purple-500 hover:underline">
                                    <Linkedin className="h-3.5 w-3.5 shrink-0" />
                                    rohit-krishnan-s
                                </a>
                                <a href="https://x.com/KrishrSomu" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-sm text-purple-500 hover:underline">
                                    <Twitter className="h-3.5 w-3.5 shrink-0" />
                                    KrishrSomu
                                </a>
                            </div>
                        </div>

                        {/* Education cards */}
                        <div className="w-full space-y-3">
                            <h3 className="text-sm font-semibold">Education</h3>
                            <div className="w-full rounded-lg border border-border p-4 space-y-1">
                                <div className="text-sm font-medium">Simon Fraser University</div>
                                <div className="text-xs text-purple-500">M.S. Computer Science</div>
                                <div className="text-xs text-muted-foreground">Visual Computing · Sep 2019 – Apr 2021</div>
                                <div className="text-xs text-muted-foreground">Burnaby, Canada</div>
                            </div>
                            <div className="w-full rounded-lg border border-border p-4 space-y-1">
                                <div className="text-sm font-medium">Sairam Engineering College</div>
                                <div className="text-xs text-purple-500">B.S. Computer Science</div>
                                <div className="text-xs text-muted-foreground">Aug 2014 – Apr 2018 · Chennai, India</div>
                            </div>
                        </div>

                        {/* Skills strip */}
                        <div className="w-full space-y-3">
                            <h3 className="text-sm font-semibold">Skills &amp; Stack</h3>
                            <div className="flex flex-wrap gap-2">
                                {skills.map(skill => (
                                    <span key={skill} className="text-xs px-2.5 py-1 rounded-md bg-muted border border-border text-muted-foreground">
                                        {skill}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Beyond the Work — full width */}
                <div className="rounded-lg border border-border bg-muted/30 p-5">
                    <h3 className="text-sm font-semibold mb-3">Beyond the Work</h3>
                    <Text size="sm" variant="muted">
                        What drives me isn&apos;t just the engineering — it&apos;s the questions underneath. How does intelligence arise?
                        What does it mean for a system to truly understand something? These questions pull me as much toward
                        cosmology and the origins of life as toward ML research. I find the boundary between physics,
                        biology, and computation endlessly fascinating.
                    </Text>
                    <Text size="sm" variant="muted" className="mt-3">
                        This site is where I think out loud — about ML, math, and ideas I&apos;m working through.
                        If any of it resonates, I&apos;d love to hear from you.
                    </Text>
                </div>
            </StackVertical>
            <SectionFooter color="purple" />
        </BaseContainer>
    )
}
