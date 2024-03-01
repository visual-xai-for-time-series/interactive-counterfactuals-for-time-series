import { Injectable } from '@angular/core'

@Injectable({
    providedIn: 'root',
})
export class DataService {
    private counter: number = 0

    constructor() {}

    getCounter(): any {
        return this.counter
    }

    addToCounter(add: number = 1): void {
        this.counter += add
    }
}
