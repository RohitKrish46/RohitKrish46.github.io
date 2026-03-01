import { motion } from 'framer-motion'
import { StackVertical } from '@/components/layout/layout-stack/layout-stack'
import Text from '@/components/ui/text/text'
import TextHeading from '@/components/ui/text-heading/text-heading'

interface EducationItemProps {
    degree: string;
    field: string;
    institution: string;
    period: string;
    delay: number;
}

function EducationItem({ degree, field, institution, period, delay }: EducationItemProps) {
    return (
        <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay, duration: 0.4 }}
            className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-0.5 sm:gap-4"
        >
            <div>
                <Text size="sm">
                    <span className="font-medium">{degree}</span>
                    {', '}
                    <span className="text-purple-500">{field}</span>
                    {' · '}
                    <span>{institution}</span>
                </Text>
            </div>
            <Text size="sm" variant="muted" className="flex-shrink-0 text-xs sm:text-sm">
                {period}
            </Text>
        </motion.div>
    )
}

export function Education() {
    const items = [
        {
            degree: "MS, Computer Science",
            field: "Visual Computing",
            institution: "Simon Fraser University",
            period: "2019 – 2021",
        },
        {
            degree: "BE, Computer Science",
            field: "Computer Science",
            institution: "Sairam Engineering College",
            period: "2014 – 2018",
        },
    ]

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9, duration: 0.5 }}
        >
            <StackVertical gap="sm">
                <TextHeading as="h2">Education</TextHeading>
                <StackVertical gap="sm">
                    {items.map((item, index) => (
                        <EducationItem
                            key={index}
                            {...item}
                            delay={1.0 + index * 0.1}
                        />
                    ))}
                </StackVertical>
            </StackVertical>
        </motion.div>
    )
}
