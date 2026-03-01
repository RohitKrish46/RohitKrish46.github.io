import { motion } from 'framer-motion'
import { StackVertical } from '@/components/layout/layout-stack/layout-stack'
import Text from '@/components/ui/text/text'
import TextHeading from '@/components/ui/text-heading/text-heading'
import Link from 'next/link'

interface WorkItemProps {
    role: string;
    company: string;
    companyUrl?: string;
    period: string;
    delay: number;
}

function WorkItem({ role, company, companyUrl, period, delay }: WorkItemProps) {
    return (
        <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay, duration: 0.4 }}
            className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-0.5 sm:gap-4"
        >
            <div>
                <Text size="sm">
                    <span className="font-medium">{role}</span>
                    {' · '}
                    {companyUrl ? (
                        <Link
                            href={companyUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-purple-500 hover:underline"
                        >
                            {company}
                        </Link>
                    ) : (
                        <span>{company}</span>
                    )}
                </Text>
            </div>
            <Text size="sm" variant="muted" className="flex-shrink-0 text-xs sm:text-sm">
                {period}
            </Text>
        </motion.div>
    )
}

export function WorkExperience() {
    const items = [
        {
            role: "Machine Learning Engineer III, MLOps",
            company: "ACV Auctions",
            period: "Sep 2025 – Present",
        },
        {
            role: "Machine Learning Engineer",
            company: "AmiableAi Inc.",
            period: "Jan 2023 – Oct 2023",
        },
        {
            role: "Machine Learning Engineer",
            company: "VisionBox Inc.",
            period: "Mar 2022 – Dec 2022",
        },
        {
            role: "Research Intern",
            company: "Medical Image Analysis Lab, SFU",
            period: "May 2020 – Aug 2020",
        },
        {
            role: "Associate Software Engineer",
            company: "Prodapt Solutions",
            period: "May 2018 – Jul 2019",
        },
    ]

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.5 }}
        >
            <StackVertical gap="sm">
                <TextHeading as="h2">Work Experience</TextHeading>
                <StackVertical gap="sm">
                    {items.map((item, index) => (
                        <WorkItem
                            key={index}
                            {...item}
                            delay={0.7 + index * 0.1}
                        />
                    ))}
                </StackVertical>
            </StackVertical>
        </motion.div>
    )
}
