import { Component, OnInit, Input, ViewChild, ElementRef, AfterViewInit } from '@angular/core'
import { CommonModule } from '@angular/common'
import { InteractionsService } from '../interactions.service'
import { DataService } from '../data.service'

import * as d3 from 'd3'
import { HttpService } from '../http.service'

@Component({
    selector: 'app-scatter-plot',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './scatter-plot.component.html',
    styleUrl: './scatter-plot.component.scss',
})
export class ScatterPlotComponent implements AfterViewInit {
    @Input() settings

    @ViewChild('div') div_element: ElementRef<HTMLDivElement> | undefined

    private counterCF = 0

    private svg
    private density
    private scatter

    private dataUrl
    private densityUrl
    private projectUrl
    private inverseProjectUrl

    private labels
    private predictions
    private prediction_probabilities

    private color

    private xScale: any
    private yScale: any

    private dragHandler

    private containerWidth
    private containerHeight
    private margin = { top: 10, right: 10, bottom: 10, left: 10 }

    private minDrag = { x: 15, y: 15 }
    private minClick = { x: 1, y: 1 }

    private radius = 4

    public title: string = ''

    constructor(
        private httpService: HttpService,
        private interactionsService: InteractionsService,
        private dataService: DataService
    ) {}

    ngAfterViewInit(): void {
        this.title = this.settings[0]
        this.dataUrl = this.settings[1]
        this.densityUrl = this.settings[2]
        this.projectUrl = this.settings[3]
        this.inverseProjectUrl = this.settings[4]

        const data = this.interactionsService.getData()
        this.labels = data[1]
        this.predictions = data[2]
        this.prediction_probabilities = data[3]

        this.color = this.predictions

        this.interactionsService.getHighlightData.subscribe((data) => {
            if (data != null) {
                this.highlightData(data)
            } else {
                this.unHighlightData()
            }
        })

        this.interactionsService.getColorMode.subscribe((mode) => {
            if (mode != null) {
                console.log(mode)

                this.changeColorOfPoints(mode)
            }
        })

        this.interactionsService.getScatterData.subscribe((data) => {
            if (data != null) {
                this.projectDataAndAddToScatterPlot(this.projectUrl, data)
            } else {
                this.clearNewAddedData()
            }
        })

        this.fetchDataAndAddToScatterPlot()
    }

    private createSVG(): void {
        this.containerWidth = this.div_element?.nativeElement.clientWidth ?? 600 - 10
        this.containerHeight = (this.div_element?.nativeElement.clientWidth ?? 600) / 1.5 - 10

        this.svg = d3
            .select(this.div_element?.nativeElement)
            .append('svg')
            .attr('width', this.containerWidth)
            .attr('height', this.containerHeight)
            .style('border', 'solid black 1px')

        this.density = this.svg
            .append('g')
            .attr('transform', 'translate(' + this.margin.left + ',' + this.margin.top + ')')
        this.scatter = this.svg
            .append('g')
            .attr('transform', 'translate(' + this.margin.left + ',' + this.margin.top + ')')

        const labels = this.prediction_probabilities[0].length

        this.svg
            .append('defs')
            .append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 10)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('class', 'arrowHead')
            .style('fill', 'gray')

        Array.from({ length: labels }, (_, index) => index).forEach((value) => {
            this.svg
                .append('defs')
                .append('marker')
                .attr('id', `arrow-${value}`)
                .attr('viewBox', '0 -5 10 10')
                .attr('refX', 10)
                .attr('refY', 0)
                .attr('markerWidth', 6)
                .attr('markerHeight', 6)
                .attr('orient', 'auto')
                .append('path')
                .attr('d', 'M0,-5L10,0L0,5')
                .attr('class', 'arrowHead')
                .style('fill', d3.schemeDark2[value])
        })
    }

    private drawDensityPlot(data: any[]): void {
        if (data.length < 1) {
            return
        }

        const coordinate_data = data[0]
        const color_data = data[1]

        const length = Math.sqrt(data[0].length)

        const width = this.containerWidth - this.margin.left - this.margin.right
        const height = this.containerHeight - this.margin.top - this.margin.bottom

        const ratio_width = width / length
        const ratio_height = height / length

        const that = this
        this.density
            .selectAll('rect')
            .data(coordinate_data)
            .enter()
            .append('rect')
            .attr('x', function (d) {
                return that.xScale(d[0])
            })
            .attr('y', function (d) {
                return that.yScale(d[1])
            })
            .attr('width', ratio_width)
            .attr('height', ratio_height)
            .attr('opacity', 0.5)
            .attr('fill', function (_, i) {
                return d3.color(`rgb(${color_data[i]})`)
            })
    }

    private drawScatterPlot(data: any[]): void {
        this.xScale = d3
            .scaleLinear()
            .domain(d3.extent(data, (d) => d[0]))
            .range([0, this.containerWidth - this.margin.right - this.margin.left])

        this.yScale = d3
            .scaleLinear()
            .domain(d3.extent(data, (d) => d[1]))
            .range([this.containerHeight - this.margin.top - this.margin.bottom, 0])

        let wasDragged = false
        const that = this
        this.dragHandler = d3
            .drag()
            .clickDistance(Math.min(that.minDrag.x, that.minDrag.y))
            .on('start', function (this: any, d: any) {
                console.log('dragstart')

                const orgPoint = d3.select(this)
                const clonedPoint = orgPoint.clone()
                that.scatter.node().appendChild(clonedPoint.node())
                const idx = orgPoint.attr('data-idx')
                if (orgPoint.attr('data-idx')) {
                    orgPoint.attr('org-data-idx', idx)
                }
                orgPoint.attr('data-idx', `cf-${idx}`).attr('previous-data-idx', idx)

                wasDragged = false
            })
            .on('drag', function (this: any, d: any) {
                console.log('dragged')
                wasDragged = true
                d3.select(this).attr('cx', d.x).attr('cy', d.y)
            })
            .on('end', function (this: any, d: any) {
                console.log('dragend')

                const orgPoint = d3.select(this)
                const referenceId = orgPoint.attr('data-idx')
                const orgReferenceId = orgPoint.attr('previous-data-idx')
                const clonedPoint = that.svg.selectAll('.data-point').filter(function (this: any) {
                    return d3.select(this).attr('data-idx') === orgReferenceId
                })

                if (
                    Math.abs(orgPoint.attr('cx') - clonedPoint.attr('cx')) < that.minDrag.x &&
                    Math.abs(orgPoint.attr('cy') - clonedPoint.attr('cy')) < that.minDrag.y
                ) {
                    const [x, y] = [clonedPoint.attr('cx'), clonedPoint.attr('cy')]
                    orgPoint.attr('cx', x).attr('cy', y)
                    orgPoint.attr('data-idx', orgPoint.attr('previous-data-idx'))
                    clonedPoint.remove()
                    wasDragged = false
                }

                if (wasDragged) {
                    // Add connecting line
                    const layer = that.scatter
                        .append('g')
                        .attr('class', 'counterfactual-layer')
                        .attr('data-idx', referenceId)
                    const removedPoint = clonedPoint.remove()
                    layer.node().appendChild(removedPoint.node())

                    const endX = orgPoint.attr('cx')
                    const endY = orgPoint.attr('cy')

                    const startX = clonedPoint.attr('cx')
                    const startY = clonedPoint.attr('cy')

                    const cf_data_line = layer
                        .append('line')
                        .attr('class', 'cf-data-line')
                        .attr('stroke', 'gray')
                        .attr('stroke-width', 2)
                        .style('opacity', 0.6)
                        .attr('marker-end', 'url(#arrow)')
                        .attr('x1', startX)
                        .attr('y1', startY)
                        .attr('x2', endX)
                        .attr('y2', endY)

                    clonedPoint.raise()
                    orgPoint.raise()

                    // Predict new
                    const callback = (data) => {
                        const prediction = parseInt(data['prediction'][0])
                        const orgColor = orgPoint.attr('fill')
                        orgPoint.attr('org-fill', orgPoint.attr('fill'))
                        const newColor = d3.schemeDark2[prediction]
                        orgPoint.attr('fill', newColor).raise()

                        // Set the gradient
                        layer
                            .append('linearGradient')
                            .attr('id', `line-gradient-${referenceId}`)
                            .attr('gradientUnits', 'userSpaceOnUse')
                            .attr('x1', startX)
                            .attr('y1', startY)
                            .attr('x2', endX)
                            .attr('y2', endY)
                            .selectAll('stop')
                            .data([
                                { offset: '0%', color: orgColor },
                                { offset: '100%', color: newColor },
                            ])
                            .enter()
                            .append('stop')
                            .attr('offset', function (d) {
                                return d.offset
                            })
                            .attr('stop-color', function (d) {
                                return d.color
                            })

                        cf_data_line
                            .attr('stroke', `url(#line-gradient-${referenceId})`)
                            .attr('marker-end', `url(#arrow-${prediction})`)
                    }

                    const data_to_send = [[that.xScale.invert(endX), that.yScale.invert(endY)]]
                    that.projectDataAndAddToLinePlot(that.inverseProjectUrl, data_to_send, referenceId, true, callback)
                } else {
                    console.log('clicked')

                    d.sourceEvent.stopPropagation()

                    const idx = d3.select(this).attr('data-idx')
                    that.interactionsService.addLineData(idx)
                }
            })

        const circles = this.scatter
            .append('g')
            .attr('class', 'data-points')
            .selectAll('.data-point')
            .data(data)
            .enter()
            .append('circle')
            .attr('class', 'data-point')
            .attr('data-idx', (_, i) => i)
            .attr('cx', (d: any) => this.xScale(d[0]))
            .attr('cy', (d: any) => this.yScale(d[1]))
            .attr('r', this.radius)
            .attr('fill', (_, i: number) => d3.schemeDark2[this.color[i]])
            .on('mouseover', (d: any) => {
                const idx = d3.select(d.target).attr('data-idx')
                this.interactionsService.setHighlightData(idx)
            })
            .on('mouseout', () => {
                this.interactionsService.setHighlightData(null)
            })
            .call(this.dragHandler)
    }

    private interactionHandler(): void {
        const that = this
        this.svg.on('click', (d: any) => {
            const [x, y] = d3.pointer(d)

            const nearPoints = this.svg.selectAll('.data-point').filter(function (this: any) {
                return (
                    Math.abs(d3.select(this).attr('cx') - x) < that.minClick.x &&
                    Math.abs(d3.select(this).attr('cy') - y) < that.minClick.y
                )
            })
            if (nearPoints.size() > 0) {
                return
            }

            this.dataService.addToCounter()
            const referenceId = `self-${this.dataService.getCounter()}`

            const data_to_send = [[this.xScale.invert(x), this.yScale.invert(y)]]

            this.svg
                .append('circle')
                .attr('class', 'data-point')
                .attr('data-idx', referenceId)
                .attr('cx', x)
                .attr('cy', y)
                .attr('r', this.radius)
                .attr('fill', 'black')
                .on('mouseover', (d: any) => {
                    const idx = d3.select(d.target).attr('data-idx')
                    this.interactionsService.setHighlightData(idx)
                    d3.select(d.target).raise()
                })
                .on('mouseout', (d: any) => {
                    this.interactionsService.setHighlightData(null)
                    d3.select(d.target).raise()
                })

            this.projectDataAndAddToLinePlot(this.inverseProjectUrl, data_to_send, referenceId, true)
        })
    }

    private addNewDataPoint(newDataPoint: any, baseData: any): void {
        console.log(newDataPoint)
        console.log(baseData)

        const referenceId = baseData[0]
        const alreadyThereCheck = this.svg.selectAll('.data-point').filter(function (this: any) {
            return d3.select(this).attr('data-idx') === referenceId
        })

        if (alreadyThereCheck.size() > 0) {
            console.log(alreadyThereCheck)
            return
        }

        const dataPoint = newDataPoint['data'][0]
        const prediction = newDataPoint['prediction'][0]

        const endX = this.xScale(dataPoint[0])
        const endY = this.yScale(dataPoint[1])

        let newReferenceId = referenceId
        let symbol = d3.symbol(d3.symbolDiamond)
        const layer = this.scatter.append('g')
        if (referenceId.includes('self')) {
            layer.attr('class', 'activation-layer').attr('data-idx', referenceId)
            symbol = d3.symbol(d3.symbolStar)
        } else if (referenceId.includes('cf')) {
            const orgPoint = this.svg.selectAll('.data-point').filter(function (this: any) {
                return d3.select(this).attr('data-idx') === referenceId.slice(3)
            })

            const orgReferenceId = orgPoint.attr('data-idx')
            layer.attr('class', 'cf-layer').attr('data-idx', orgReferenceId)

            const startX = orgPoint.attr('cx')
            const startY = orgPoint.attr('cy')

            layer
                .append('line')
                .attr('class', 'cf-line')
                .attr('stroke', 'gray')
                .attr('stroke-width', 3)
                .attr('x1', startX)
                .attr('y1', startY)
                .attr('x2', endX)
                .attr('y2', endY)
        } else {
            this.counterCF += 1
            newReferenceId = `cf-${referenceId}-${this.counterCF}`

            layer.attr('class', 'cf-layer').attr('data-idx', newReferenceId)

            const oldPoint = this.svg
                .selectAll('.data-point')
                .filter(function (this: any) {
                    return d3.select(this).attr('data-idx') === referenceId
                })
                .clone()

            const startX = d3.select(oldPoint.node()).attr('cx')
            const startY = d3.select(oldPoint.node()).attr('cy')

            layer
                .append('line')
                .attr('class', 'cf-data-line')
                .attr('stroke', 'gray')
                .attr('stroke-width', 3)
                .attr('x1', startX)
                .attr('y1', startY)
                .attr('x2', endX)
                .attr('y2', endY)
        }

        layer
            .append('path')
            .attr('class', 'data-point addedPoint')
            .attr('data-idx', newReferenceId)
            .attr('transform', `translate(${endX},${endY})`)
            .attr('cx', endX)
            .attr('cy', endY)
            .attr('d', symbol)
            .attr('fill', d3.schemeDark2[prediction])
            .on('click', (d: any) => {
                const idx = d3.select(d.target).attr('data-idx')
                this.interactionsService.addLineData(idx)
            })
            .on('mouseover', (d: any) => {
                const idx = d3.select(d.target).attr('data-idx')
                this.interactionsService.setHighlightData(idx)
                d3.select(d.target).raise()
            })
            .on('mouseout', (d: any) => {
                this.interactionsService.setHighlightData(null)
                d3.select(d.target).raise()
            })
    }

    private highlightData(filterData: any): void {
        if (this.svg) {
            this.svg
                .selectAll('.data-point')
                .filter(function (this: any) {
                    return d3.select(this).attr('data-idx') === filterData
                })
                .attr('stroke', 'black')
                .attr('stroke-width', 4)
                .raise()
        }
    }

    private unHighlightData(): void {
        if (this.svg) {
            this.svg.selectAll('.data-point').attr('stroke', null).attr('stroke-width', null)
        }
    }

    private changeColorOfPoints(mode: number = 1): void {
        if (mode === 1) {
            this.color = this.labels
        } else if (mode === 2) {
            this.color = this.predictions
        }
        if (this.svg) {
            this.svg.selectAll('.data-point').attr('fill', (_, i: number) => d3.schemeDark2[this.color[i]])
        }
    }

    private clearNewAddedData(): void {
        if (this.svg) {
            this.svg.selectAll('circle').remove()

            this.svg.selectAll('.addedPoint').remove()
            this.svg.selectAll('.counterfactual-layer').remove()
        }
    }

    private fetchDataAndAddToScatterPlot(): void {
        console.log('Initalize SVG')
        this.createSVG()

        const dataPromise = this.httpService.get<any>(this.dataUrl)
        const densityPromise = this.httpService.get<any>(this.densityUrl)
        dataPromise.subscribe((data: any) => {
            const receivedData = JSON.parse(data)
            this.drawScatterPlot(receivedData['data'])

            densityPromise.subscribe((data: any) => {
                const receivedData = JSON.parse(data)
                this.drawDensityPlot(receivedData['data'])

                this.interactionHandler()
            })
        })
    }

    private projectDataAndAddToLinePlot(
        url: string,
        dataToSend: any,
        referenceId: string = '-1',
        createNewPoint: boolean = true,
        callbackFn: (data: any) => void = () => {}
    ): any {
        const dataLoading = this.httpService.post<any>(url, { data: dataToSend })
        dataLoading.subscribe((data: any) => {
            const parsedData = JSON.parse(data)
            console.log(parsedData)
            const lineData = parsedData['data'][0]
            let prediction = -1
            if ('prediction' in parsedData) {
                prediction = parseInt(parsedData['prediction'][0])
            }
            if (createNewPoint) {
                this.interactionsService.addNewScatterData([referenceId, lineData, prediction, prediction])
            }
            this.interactionsService.addNewLineData([referenceId, lineData, prediction, prediction])

            callbackFn(parsedData)
        })
    }

    private projectDataAndAddToScatterPlot(url: string, baseData: any): void {
        const dataToSend = [baseData[1]]
        console.log(dataToSend)
        this.httpService.post<any>(url, { data: dataToSend }).subscribe((data: any) => {
            const parsedData = JSON.parse(data)
            this.addNewDataPoint(parsedData, baseData)
        })
    }
}
